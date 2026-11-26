"""RELAY as an MCP server: memory across an agent's sessions.

Run it over stdio:

    relay-mcp                     # console script
    python -m context_handover.mcp.server

Wombat (or any MCP host) then sees these tools as ``mcp__relay__<tool>``:

    relay_remember      declare one atom: decision, constraint, task, ...
    relay_capture       extract atoms from transcript text
    relay_atoms         list what is currently remembered
    relay_handover      pack the context block for the next session
    relay_drift         how far this context has moved since the last handover
    relay_losses        what the last handover dropped, by name
    relay_new_session   close the session and start the next one
    relay_status        where state lives and what is in it

Every tool returns JSON text, so it is readable in a terminal and parseable by
an agent. ``relay_handover`` returns the context block itself, because that is
the thing the agent should paste into its prompt.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

from mcp.server.mcpserver import MCPServer

from .store import DEFAULT_MAX_TOKENS, RelayStore, StoreError, state_dir

logger = logging.getLogger("context_handover.mcp")

INSTRUCTIONS = """\
RELAY keeps typed memory across agent sessions.

Write to it when something must survive the context window: a decision that was
made, a constraint that must hold, a task that is still open, a question that is
unresolved. Do not capture small talk or tool output.

Read from it at the start of a session: call relay_handover, and put the returned
block in your working context. Call relay_new_session when the current piece of
work is finished, so the next session starts from the packed atoms instead of a
transcript.
"""


def build_server(store: Optional[RelayStore] = None) -> MCPServer:
    """Construct the server. ``store`` is injectable so tests do not need disk."""
    store = store or RelayStore()
    server = MCPServer(
        name="relay",
        title="RELAY context handover",
        version="0.2.0",
        instructions=INSTRUCTIONS,
    )

    def dump(payload) -> str:
        return json.dumps(payload, indent=2, sort_keys=True, default=str)

    @server.tool()
    def relay_remember(kind: str, text: str, confidence: float = 0.8) -> str:
        """Remember one fact that must survive this session.

        Args:
            kind: One of decision, constraint, task, question, entity, belief,
                relation. Decisions and constraints are never dropped by the
                packer, so use them for anything that must hold.
            text: The assertion, in one sentence, as a standalone statement.
            confidence: How sure you are, 0 to 1. Default 0.8.
        """
        try:
            record, created = store.remember(kind, text, confidence)
        except (StoreError, ValueError) as exc:
            return dump({"ok": False, "error": str(exc)})
        store.save()
        return dump(
            {
                "ok": True,
                "created": created,
                "atom_id": record.atom_id,
                "type": record.atom_type,
                "content": record.content,
                "confidence": round(record.confidence, 2),
                "session_id": store.session_id,
            }
        )

    @server.tool()
    def relay_capture(messages: list[str], role: str = "user") -> str:
        """Extract atoms from raw transcript text and remember them.

        Uses the model when OPENAI_API_KEY is set, otherwise a regex fallback
        that keeps only explicit decisions, constraints, tasks and questions.
        Prefer relay_remember when you already know what to store.

        Args:
            messages: Transcript lines or turns to mine for atoms.
            role: Label for the speaker, for provenance. Default "user".
        """
        try:
            result = store.capture(messages, role=role)
        except StoreError as exc:
            return dump({"ok": False, "error": str(exc)})
        return dump({"ok": True, **result})

    @server.tool()
    def relay_atoms(kind: Optional[str] = None, include_resolved: bool = False) -> str:
        """List what is currently remembered, newest last.

        Args:
            kind: Filter to one atom type, for example "constraint".
            include_resolved: Include atoms that were superseded or resolved.
        """
        records = store.atoms(include_resolved=include_resolved)
        if kind:
            records = [r for r in records if r.atom_type == kind.strip().lower()]
        return dump(
            {
                "ok": True,
                "count": len(records),
                "session_id": store.session_id,
                "atoms": [
                    {
                        "atom_id": r.atom_id,
                        "type": r.atom_type,
                        "content": r.content,
                        "confidence": round(r.confidence, 2),
                        "status": r.status,
                    }
                    for r in records
                ],
            }
        )

    @server.tool()
    def relay_handover(max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Pack the context atoms into a block for the next session.

        Decisions and constraints are always included; everything else is
        selected to maximise value inside the token budget, and what was left
        out is recorded so it can be reported.

        Args:
            max_tokens: Token budget for the returned block. Default 2000.
        """
        if max_tokens < 50:
            return dump({"ok": False, "error": "max_tokens must be at least 50"})
        package = store.handover(max_tokens=max_tokens)
        settlement = store.settle_handover(package)
        return dump(
            {
                "ok": True,
                "session_id": store.session_id,
                "selected": len(package.selected_atoms),
                "total_atoms": len(store.atoms()),
                "tokens": package.total_tokens,
                "max_tokens": max_tokens,
                "dropped": settlement["dropped"],
                "context": package.to_context_string(),
            }
        )

    @server.tool()
    def relay_drift() -> str:
        """Report how far the current context has moved since the last handover."""
        return dump({"ok": True, **store.drift()})

    @server.tool()
    def relay_new_session() -> str:
        """End this session and start the next one with a packed handover.

        The returned context block is what the new session should start from.
        """
        package = store.handover(max_tokens=DEFAULT_MAX_TOKENS)
        store.settle_handover(package)
        context = package.to_context_string()
        previous, current = store.new_session()
        return dump(
            {
                "ok": True,
                "previous_session": previous,
                "session_id": current,
                "carried_atoms": len(package.selected_atoms),
                "tokens": package.total_tokens,
                "context": context,
            }
        )

    @server.tool()
    def relay_losses() -> str:
        """What the last handover left behind: every atom that did not fit.

        This is the point of the project. A summary cannot tell you what it
        dropped; this can, by name and by type.
        """
        return dump({"ok": True, **store.last_losses()})

    @server.tool()
    def relay_status() -> str:
        """Where RELAY keeps its state and what is in it."""
        return dump({"ok": True, **store.status()})

    return server


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="relay-mcp",
        description="RELAY context handover as an MCP stdio server",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help=f"where to keep state (default: RELAY_STATE_DIR or {state_dir()})",
    )
    parser.add_argument("--namespace", default=None, help="workspace namespace")
    parser.add_argument("--check", action="store_true", help="print the tool list and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        stream=sys.stderr,
    )

    store = RelayStore(directory=args.state_dir, namespace=args.namespace)
    server = build_server(store)

    if args.check:
        print(f"relay-mcp 0.2.0  namespace={store.namespace}")
        print(f"state: {store.path}")
        print("tools:")
        for tool in sorted(_tool_names(server)):
            print(f"  - {tool}")
        return 0

    server.run("stdio")
    return 0


def _tool_names(server: MCPServer) -> list[str]:
    import anyio

    tools = anyio.run(server.list_tools)
    return sorted(tool.name for tool in tools)


if __name__ == "__main__":
    raise SystemExit(main())
