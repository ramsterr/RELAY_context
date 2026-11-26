"""The plugin surface: store semantics and the seven MCP tools."""
from __future__ import annotations

import json

import pytest

from context_handover.mcp.server import build_server
from context_handover.mcp.store import RelayStore, StoreError


@pytest.fixture()
def store(tmp_path) -> RelayStore:
    return RelayStore(directory=tmp_path, namespace="test")


def test_remember_persists_across_instances(tmp_path):
    first = RelayStore(directory=tmp_path, namespace="test")
    first.remember("decision", "Use RS256 for signing tokens")
    first.save()

    second = RelayStore(directory=tmp_path, namespace="test")
    atoms = second.atoms()
    assert len(atoms) == 1
    assert atoms[0].content == "Use RS256 for signing tokens"


def test_same_canonical_form_updates_instead_of_duplicating(store):
    _, created_first = store.remember("decision", "Use RS256 for signing")
    _, created_second = store.remember("decision", "use rs256 for signing")

    assert created_first is True
    assert created_second is False
    assert len(store.atoms()) == 1
    assert store.atoms()[0].confidence > 0.8


def test_different_types_do_not_collide(store):
    store.remember("task", "rotate the keys")
    store.remember("constraint", "rotate the keys")
    assert len(store.atoms()) == 2


def test_empty_text_is_rejected(store):
    with pytest.raises(StoreError):
        store.remember("decision", "   ")


def test_handover_keeps_decisions_and_constraints_under_pressure(store):
    for i in range(40):
        store.remember("entity", f"entity number {i} with padding words here")
    store.remember("constraint", "must use RS256")

    package = store.handover(max_tokens=120)
    contents = [atom.content for atom in package.selected_atoms]

    assert "must use RS256" in contents
    assert package.total_tokens <= 120


def test_handover_reports_what_it_dropped(store):
    for i in range(30):
        store.remember("entity", f"entity {i} " + "pad " * 8)
    package = store.handover(max_tokens=80)
    settlement = store.settle_handover(package)

    assert settlement["dropped_count"] > 0
    assert len(settlement["dropped"]) == settlement["dropped_count"]


def test_handover_marks_survivors_so_they_rank_higher_next_time(store):
    store.remember("entity", "a short fact")
    store.remember("entity", "b " + "pad " * 40)
    first = store.handover(max_tokens=60)
    store.settle_handover(first)

    survivor = [a for a in first.selected_atoms][0].atom_id
    raw = store._state["atoms"][survivor]
    assert raw["handover_count"] >= 1


def test_drift_compares_against_the_previous_snapshot(store):
    store.remember("decision", "use RS256")
    package = store.handover(max_tokens=200)
    store.settle_handover(package)

    same = store.drift()
    assert same["compared_to_snapshot"] is True
    assert same["jaccard"] == 1.0

    store.remember("task", "write middleware")
    moved = store.drift()
    assert moved["jaccard"] < 1.0


def test_drift_with_no_atoms_is_explicit(store):
    assert store.drift()["verdict"] == "NO_ATOMS"


def test_new_session_keeps_atoms_and_records_the_parent(store):
    store.remember("decision", "use RS256")
    previous, current = store.new_session()

    assert previous != current
    assert store.session["parents"] == [previous]
    assert len(store.atoms()) == 1
    assert store.status()["sessions_archived"] == 1


def test_namespaces_do_not_share_memory(tmp_path):
    one = RelayStore(directory=tmp_path, namespace="one")
    one.remember("decision", "only in one")
    one.save()

    two = RelayStore(directory=tmp_path, namespace="two")
    assert two.atoms() == []


def test_corrupt_state_file_fails_loudly(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StoreError):
        RelayStore(directory=tmp_path, namespace="broken")


def test_server_registers_every_tool(store):
    import anyio

    server = build_server(store)
    tools = anyio.run(server.list_tools)
    names = sorted(tool.name for tool in tools)
    assert names == [
        "relay_atoms",
        "relay_capture",
        "relay_drift",
        "relay_handover",
        "relay_losses",
        "relay_new_session",
        "relay_remember",
        "relay_status",
    ]


def test_tool_remember_rejects_unknown_kind(store):
    import anyio

    server = build_server(store)
    result = anyio.run(
        server.call_tool, "relay_remember", {"kind": "nonsense", "text": "x"}
    )
    payload = json.loads(result.content[0].text)
    assert payload["ok"] is False
    assert "error" in payload


def test_tool_handover_returns_a_context_block(store):
    import anyio

    server = build_server(store)
    store.remember("decision", "use RS256")
    result = anyio.run(server.call_tool, "relay_handover", {"max_tokens": 200})
    payload = json.loads(result.content[0].text)

    assert payload["ok"] is True
    assert "[decision] use RS256" in payload["context"]
