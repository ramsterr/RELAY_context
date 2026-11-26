#!/usr/bin/env python3
"""End-to-end check: spawn relay-mcp over stdio and drive it like a host would.

This is the same path Wombat takes: a child process speaking JSON-RPC, tool
discovery, then tool calls.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


def payload(result) -> dict:
    text = "".join(block.text for block in result.content if block.type == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


async def main() -> int:
    state_dir = Path(tempfile.mkdtemp(prefix="relay-smoke-"))
    env = {
        **os.environ,
        "RELAY_STATE_DIR": str(state_dir),
        "RELAY_NAMESPACE": "smoke",
        "PYTHONPATH": str(ROOT / "src"),
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "context_handover.mcp.server"],
        env=env,
        cwd=str(ROOT),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(tool.name for tool in tools.tools)
            print("tools:", ", ".join(names))

            remembered = payload(
                await session.call_tool(
                    "relay_remember",
                    {"kind": "decision", "text": "Use RS256 for signing tokens"},
                )
            )
            print("remember:", remembered)
            assert remembered["ok"] and remembered["created"]

            await session.call_tool(
                "relay_remember",
                {"kind": "constraint", "text": "Tokens must expire after one hour"},
            )
            await session.call_tool(
                "relay_remember",
                {"kind": "task", "text": "Write JWT validation middleware"},
            )

            handover = payload(await session.call_tool("relay_handover", {"max_tokens": 800}))
            print("handover tokens:", handover["tokens"], "selected:", handover["selected"])
            print(handover["context"])
            assert "[decision] Use RS256 for signing tokens" in handover["context"]
            assert "[constraint] Tokens must expire after one hour" in handover["context"]

            drift = payload(await session.call_tool("relay_drift", {}))
            print("drift:", drift)

            status = payload(await session.call_tool("relay_status", {}))
            print("status:", status)
            assert status["atoms"] == 3

            fresh = payload(await session.call_tool("relay_new_session", {}))
            assert fresh["previous_session"] != fresh["session_id"]
            assert fresh["carried_atoms"] >= 2
            print("new session:", fresh["session_id"], "carried:", fresh["carried_atoms"])

    print("\npersisted:", sorted(p.name for p in state_dir.glob("*.json")))
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
