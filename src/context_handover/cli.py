"""Command line for RELAY, so the plugin is usable without an agent.

    relay status
    relay remember decision "Use RS256 for signing tokens"
    relay atoms --kind constraint
    relay handover --max-tokens 800
    relay drift
    relay losses
    relay new-session
    relay capture notes.md
    relay mcp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .mcp.store import DEFAULT_MAX_TOKENS, RelayStore, StoreError


def _store(args) -> RelayStore:
    return RelayStore(directory=args.state_dir, namespace=args.namespace)


def cmd_status(args) -> int:
    print(json.dumps(_store(args).status(), indent=2, sort_keys=True))
    return 0


def cmd_remember(args) -> int:
    store = _store(args)
    record, created = store.remember(args.kind, " ".join(args.text), args.confidence)
    store.save()
    print(f"{'added' if created else 'updated'}: [{record.atom_type}] {record.content}")
    return 0


def cmd_atoms(args) -> int:
    store = _store(args)
    records = store.atoms(include_resolved=args.all)
    if args.kind:
        records = [r for r in records if r.atom_type == args.kind]
    if not records:
        print("no atoms")
        return 0
    for record in records:
        print(f"{record.atom_id}  [{record.atom_type}] {record.content}")
    return 0


def cmd_handover(args) -> int:
    store = _store(args)
    package = store.handover(max_tokens=args.max_tokens)
    settlement = store.settle_handover(package)
    if args.json:
        print(
            json.dumps(
                {
                    "selected": len(package.selected_atoms),
                    "total_atoms": len(store.atoms()),
                    "tokens": package.total_tokens,
                    "dropped": settlement["dropped"],
                    "context": package.to_context_string(),
                },
                indent=2,
            )
        )
        return 0
    print(package.to_context_string())
    print(
        f"\n{len(package.selected_atoms)}/{len(store.atoms())} atoms, "
        f"{package.total_tokens}/{args.max_tokens} tokens, "
        f"{settlement['dropped_count']} dropped",
        file=sys.stderr,
    )
    return 0


def cmd_drift(args) -> int:
    print(json.dumps(_store(args).drift(), indent=2, sort_keys=True))
    return 0


def cmd_losses(args) -> int:
    print(json.dumps(_store(args).last_losses(), indent=2, sort_keys=True))
    return 0


def cmd_new_session(args) -> int:
    store = _store(args)
    package = store.handover(max_tokens=args.max_tokens)
    store.settle_handover(package)
    previous, current = store.new_session()
    print(f"{previous} -> {current}")
    if args.print_context:
        print()
        print(package.to_context_string())
    return 0


def cmd_capture(args) -> int:
    store = _store(args)
    if args.path:
        text = Path(args.path).read_text(encoding="utf-8")
        messages = [line for line in text.splitlines() if line.strip()]
    else:
        messages = [line for line in sys.stdin.read().splitlines() if line.strip()]
    result = store.capture(messages)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_reset(args) -> int:
    if not args.yes:
        print("refusing to wipe state without --yes", file=sys.stderr)
        return 2
    store = _store(args)
    store.reset()
    print(f"reset {store.path}")
    return 0


def cmd_mcp(args) -> int:
    from .mcp.server import main as server_main

    argv = []
    if args.state_dir:
        argv += ["--state-dir", args.state_dir]
    if args.namespace:
        argv += ["--namespace", args.namespace]
    return server_main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relay", description="RELAY context handover")
    parser.add_argument("--state-dir", default=None, help="override RELAY_STATE_DIR")
    parser.add_argument("--namespace", default=None, help="workspace namespace")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="state location and contents").set_defaults(func=cmd_status)

    remember = sub.add_parser("remember", help="add one atom")
    remember.add_argument("kind", choices=["decision", "constraint", "task", "question", "entity", "belief", "relation"])
    remember.add_argument("text", nargs="+")
    remember.add_argument("--confidence", type=float, default=0.8)
    remember.set_defaults(func=cmd_remember)

    atoms = sub.add_parser("atoms", help="list atoms")
    atoms.add_argument("--kind", default=None)
    atoms.add_argument("--all", action="store_true", help="include resolved atoms")
    atoms.set_defaults(func=cmd_atoms)

    handover = sub.add_parser("handover", help="pack the context block")
    handover.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    handover.add_argument("--json", action="store_true")
    handover.set_defaults(func=cmd_handover)

    sub.add_parser("drift", help="drift since the last handover").set_defaults(func=cmd_drift)
    sub.add_parser("losses", help="what the last handover dropped").set_defaults(func=cmd_losses)

    new_session = sub.add_parser("new-session", help="close this session, start the next")
    new_session.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    new_session.add_argument("--print-context", action="store_true")
    new_session.set_defaults(func=cmd_new_session)

    capture = sub.add_parser("capture", help="extract atoms from a file or stdin")
    capture.add_argument("path", nargs="?", default=None)
    capture.set_defaults(func=cmd_capture)

    reset = sub.add_parser("reset", help="wipe state for this namespace")
    reset.add_argument("--yes", action="store_true")
    reset.set_defaults(func=cmd_reset)

    sub.add_parser("mcp", help="run the MCP stdio server").set_defaults(func=cmd_mcp)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
