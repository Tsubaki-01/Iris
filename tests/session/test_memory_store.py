from __future__ import annotations

import pytest

from iris.exceptions import IrisSessionError
from iris.session import InMemorySessionStore


def test_memory_store_appends_same_event_once() -> None:
    store = InMemorySessionStore()
    event = {"tool_name": "write_file", "result": {"path": "notes.txt"}}

    store.append_tool_event("session_1", "tool_result:run_1:call_1", event)
    store.append_tool_event(
        "session_1",
        "tool_result:run_1:call_1",
        {"result": {"path": "notes.txt"}, "tool_name": "write_file"},
    )

    assert store.load_tool_events("session_1") == [
        {
            "tool_name": "write_file",
            "result": {"path": "notes.txt"},
            "event_id": "tool_result:run_1:call_1",
        }
    ]


def test_memory_store_rejects_different_payload_for_existing_event_id() -> None:
    store = InMemorySessionStore()
    store.append_tool_event("session_1", "tool_result:run_1:call_1", {"status": "ok"})

    with pytest.raises(IrisSessionError):
        store.append_tool_event("session_1", "tool_result:run_1:call_1", {"status": "error"})


def test_memory_store_rejects_conflicting_or_non_json_idempotent_events() -> None:
    store = InMemorySessionStore()

    with pytest.raises(IrisSessionError):
        store.append_tool_event(
            "session_1",
            "tool_result:run_1:call_1",
            {"event_id": "tool_result:run_1:call_2", "status": "ok"},
        )
    with pytest.raises(IrisSessionError):
        store.append_tool_event("session_1", "tool_result:run_1:call_1", {"not_json": {"set"}})
