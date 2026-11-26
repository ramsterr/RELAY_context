"""Durable state for the RELAY plugin.

A coding agent calls these tools across a whole session, and across sessions,
so state cannot live in process memory only. Everything is one JSON file per
workspace namespace, written atomically.

Design notes that matter for the plugin:

* Dedupe is by ``(atom_type, canonical_form)``, the field the extractor prompt
  already declares unique. No embeddings are stored, so the file stays small
  and readable.
* A session is a unit of work. A handover records what was carried forward and
  what was dropped, and that record is what the drift metric compares against.
* Nothing here imports the vector-store backends, so the plugin runs with the
  core install alone.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from ..core.atoms import AtomStatus, AtomType, SemanticAtom
from ..core.budget import HandoverPackage, TokenBudgetManager

STATE_VERSION = 1
DEFAULT_MAX_TOKENS = 2000

#: A handover keeps the most recent decisions; anything older than this is a
#: historical record, not live context.
ACTIVE_STATUSES = {AtomStatus.ACTIVE.value}


class StoreError(RuntimeError):
    """Raised when the on-disk state cannot be read or written."""


def state_dir() -> Path:
    """Where the plugin keeps its state. ``RELAY_STATE_DIR`` wins."""
    configured = os.environ.get("RELAY_STATE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".relay"


def namespace_for(cwd: Optional[str] = None) -> str:
    """One namespace per workspace, so two projects do not share memory."""
    explicit = os.environ.get("RELAY_NAMESPACE")
    if explicit:
        return explicit
    root = cwd or os.getcwd()
    digest = hashlib.sha256(os.path.abspath(root).encode()).hexdigest()[:10]
    return f"{Path(root).name or 'root'}-{digest}"


def _now() -> float:
    return time.time()


@dataclass
class AtomRecord:
    atom_id: str
    atom_type: str
    content: str
    canonical_form: str
    confidence: float
    salience: float
    status: str
    created_at: float
    last_seen: float
    sessions: list[str]
    handover_count: int
    loss_events: int
    origin_session: str

    def to_atom(self) -> SemanticAtom:
        atom = SemanticAtom(
            atom_id=self.atom_id,
            atom_type=AtomType(self.atom_type),
            content=self.content,
            canonical_form=self.canonical_form,
            confidence=self.confidence,
            salience=self.salience,
            status=AtomStatus(self.status),
            sessions_present=list(self.sessions),
            handover_count=self.handover_count,
            loss_events=self.loss_events,
            origin_session=self.origin_session,
        )
        return atom

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, raw: dict) -> "AtomRecord":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in fields})


class RelayStore:
    """Read/write access to one workspace's atoms, sessions and history."""

    def __init__(self, directory: Optional[Path] = None, namespace: Optional[str] = None):
        self.dir = Path(directory) if directory else state_dir()
        self.namespace = namespace or namespace_for()
        self.path = self.dir / f"{self.namespace}.json"
        self.budget = TokenBudgetManager()
        self._state = self._load()

    # ---------------------------------------------------------------- storage

    def _empty(self) -> dict:
        return {
            "version": STATE_VERSION,
            "namespace": self.namespace,
            "created_at": _now(),
            "session": {
                "id": uuid.uuid4().hex[:8],
                "created_at": _now(),
                "parents": [],
                "messages": 0,
            },
            "sessions": [],
            "atoms": {},
            "snapshots": [],
        }

    def _load(self) -> dict:
        if not self.path.exists():
            return self._empty()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreError(f"cannot read {self.path}: {exc}") from exc
        if raw.get("version") != STATE_VERSION:
            raise StoreError(
                f"{self.path} is version {raw.get('version')}, expected {STATE_VERSION}"
            )
        return raw

    def save(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".relay-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._state, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp, self.path)
        except OSError as exc:
            os.unlink(tmp) if os.path.exists(tmp) else None
            raise StoreError(f"cannot write {self.path}: {exc}") from exc
        return self.path

    def reset(self) -> None:
        self._state = self._empty()
        self.save()

    # ---------------------------------------------------------------- session

    @property
    def session(self) -> dict:
        return self._state["session"]

    @property
    def session_id(self) -> str:
        return self.session["id"]

    def new_session(self) -> tuple[str, str]:
        previous = self.session_id
        self._state["sessions"].append(
            {
                **self.session,
                "ended_at": _now(),
                "atom_count": len(self._records()),
            }
        )
        self._state["session"] = {
            "id": uuid.uuid4().hex[:8],
            "created_at": _now(),
            "parents": [previous],
            "messages": 0,
        }
        self.save()
        return previous, self.session_id

    # ------------------------------------------------------------------ atoms

    def _records(self) -> dict[str, AtomRecord]:
        return {
            atom_id: AtomRecord.from_dict(raw)
            for atom_id, raw in self._state["atoms"].items()
        }

    def atoms(self, include_resolved: bool = False) -> list[AtomRecord]:
        records = list(self._records().values())
        if include_resolved:
            return sorted(records, key=lambda r: r.created_at)
        return sorted(
            (r for r in records if r.status == AtomStatus.ACTIVE.value),
            key=lambda r: r.created_at,
        )

    def remember(
        self,
        kind: str,
        text: str,
        confidence: float = 0.8,
        canonical_form: Optional[str] = None,
    ) -> tuple[AtomRecord, bool]:
        """Add one atom. Returns ``(record, created)``.

        The agent declaring an atom directly is more reliable than inferring one
        from prose, so this is the primary write path; extraction is the
        automatic fallback.
        """
        atom_type = AtomType(str(kind).strip().lower())
        content = " ".join(str(text).split())
        if not content:
            raise StoreError("atom text is empty")
        canonical = (canonical_form or content).strip().lower()[:200]
        atom_id = SemanticAtom.make_id(canonical, atom_type)

        records = self._state["atoms"]
        if atom_id in records:
            raw = records[atom_id]
            raw["confidence"] = min(1.0, max(raw["confidence"], float(confidence)) + 0.05)
            raw["salience"] = min(1.0, raw["salience"] + 0.1)
            raw["last_seen"] = _now()
            if self.session_id not in raw["sessions"]:
                raw["sessions"].append(self.session_id)
            return AtomRecord.from_dict(raw), False

        record = AtomRecord(
            atom_id=atom_id,
            atom_type=atom_type.value,
            content=content,
            canonical_form=canonical,
            confidence=max(0.0, min(1.0, float(confidence))),
            salience=max(0.0, min(1.0, float(confidence))),
            status=AtomStatus.ACTIVE.value,
            created_at=_now(),
            last_seen=_now(),
            sessions=[self.session_id],
            handover_count=0,
            loss_events=0,
            origin_session=self.session_id,
        )
        records[atom_id] = record.to_dict()
        return record, True

    def capture(self, messages: Iterable[str], role: str = "user") -> dict:
        """Extract atoms from raw transcript text.

        Uses the LLM extractor when one is configured, otherwise the regex
        fallback, so the plugin keeps working with no API key.
        """
        from ..extraction.extraction import AtomExtractor

        texts = [" ".join(str(message).split()) for message in messages]
        texts = [text for text in texts if text]
        if not texts:
            return {"extracted": 0, "created": 0, "updated": 0, "mode": "empty"}

        client = _openai_client()
        extractor = AtomExtractor(client)
        mode = "llm" if client else "regex"

        created = updated = 0
        seen = set()
        for text in texts:
            try:
                candidates = (
                    extractor.extract(text) if client else extractor._regex_extract(text)
                )
            except Exception:
                candidates = extractor._regex_extract(text)
                mode = "regex-fallback"
            for candidate in candidates:
                key = (candidate.type, candidate.canonical_form.lower())
                if key in seen:
                    continue
                seen.add(key)
                _, was_created = self.remember(
                    candidate.type,
                    candidate.content,
                    candidate.confidence,
                    candidate.canonical_form,
                )
                created += int(was_created)
                updated += int(not was_created)

        self.session["messages"] += len(texts)
        self.save()
        return {
            "extracted": len(seen),
            "created": created,
            "updated": updated,
            "mode": mode,
            "session_id": self.session_id,
        }

    # -------------------------------------------------------------- selection

    def handover(self, max_tokens: int = DEFAULT_MAX_TOKENS) -> HandoverPackage:
        records = self.atoms()
        atoms = [record.to_atom() for record in records]
        package = HandoverPackage(atoms, self.budget, max_tokens)
        package.build()
        return package

    def settle_handover(self, package: HandoverPackage) -> dict:
        """Record which atoms survived, so the next handover can weight them."""
        included = {atom.atom_id for atom in package.selected_atoms}
        dropped = []
        for atom_id, raw in self._state["atoms"].items():
            if raw["status"] != AtomStatus.ACTIVE.value:
                continue
            if atom_id in included:
                raw["handover_count"] += 1
            else:
                raw["loss_events"] += 1
                dropped.append(
                    {
                        "atom_id": atom_id,
                        "type": raw["atom_type"],
                        "content": raw["content"],
                    }
                )
        self._state["snapshots"].append(
            {
                "session_id": self.session_id,
                "at": _now(),
                "included": sorted(included),
                "dropped": dropped,
            }
        )
        self._state["snapshots"] = self._state["snapshots"][-50:]
        self.save()
        return {
            "included": len(included),
            "dropped": dropped,
            "dropped_count": len(dropped),
        }

    def drift(self) -> dict:
        """Compare this handover against the previous one.

        The previous snapshot is the reference, which is what makes the number
        mean something: Jaccard over atom ids that survived versus the set that
        was live last time.
        """
        from ..measurement.drift import DriftMeasurementSuite

        records = self.atoms()
        current_ids = {record.atom_id for record in records}
        snapshots = self._state["snapshots"]
        previous_ids = set(snapshots[-1]["included"]) if snapshots else set()

        suite = DriftMeasurementSuite()
        if not records:
            return {"composite": 0.0, "verdict": "NO_ATOMS", "atoms": 0}

        type_counts: dict[str, int] = {}
        for record in records:
            type_counts[record.atom_type] = type_counts.get(record.atom_type, 0) + 1
        model_dist = {k: v / len(records) for k, v in type_counts.items()}
        uniform = {t.value: 1 / len(AtomType) for t in AtomType}

        kl_structural = suite.kl_structural(
            {r.atom_id: r.to_atom() for r in records}, model_dist or uniform
        )
        jaccard = suite.jaccard(current_ids, previous_ids) if previous_ids else 1.0
        composite = suite.composite(kl_structural, jaccard)
        return {
            "kl_structural": round(float(kl_structural), 4),
            "jaccard": round(float(jaccard), 4),
            "composite": round(float(composite), 4),
            "verdict": suite.verdict(composite),
            "atoms": len(records),
            "compared_to_snapshot": bool(previous_ids),
        }

    def last_losses(self) -> dict:
        """What the most recent handover left behind, and why."""
        snapshots = self._state["snapshots"]
        if not snapshots:
            return {"handovers": 0, "dropped": [], "dropped_count": 0}
        last = snapshots[-1]
        dropped = last.get("dropped", [])
        return {
            "handovers": len(snapshots),
            "session_id": last.get("session_id"),
            "at": last.get("at"),
            "included": len(last.get("included", [])),
            "dropped": dropped,
            "dropped_count": len(dropped),
        }

    def status(self) -> dict:
        records = self.atoms()
        by_type: dict[str, int] = {}
        for record in records:
            by_type[record.atom_type] = by_type.get(record.atom_type, 0) + 1
        return {
            "namespace": self.namespace,
            "state_file": str(self.path),
            "session_id": self.session_id,
            "sessions_archived": len(self._state["sessions"]),
            "atoms": len(records),
            "by_type": by_type,
            "handovers": len(self._state["snapshots"]),
            "messages_seen": self.session["messages"],
            "parent_sessions": self.session["parents"],
        }

    def context_block(self, package: HandoverPackage) -> str:
        return package.to_context_string()


def _openai_client():
    """An OpenAI client when a key and the package are both available."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        return OpenAI()
    except Exception:
        return None
