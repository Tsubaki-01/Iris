from __future__ import annotations

import pytest

from iris.exceptions import IrisSessionError
from iris.session import InMemorySessionStore


def test_memory_store_appends_same_event_once() -> None:
    store = InMemorySessionStore()
    event = {
        "event_id": "tool_result:run_1:call_1",
        "tool_name": "write_file",
        "result": {"path": "notes.txt"},
    }

    store.append_tool_event("session_1", event)
    store.append_tool_event(
        "session_1",
        {
            "result": {"path": "notes.txt"},
            "tool_name": "write_file",
            "event_id": "tool_result:run_1:call_1",
        },
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
    store.append_tool_event(
        "session_1",
        {"event_id": "tool_result:run_1:call_1", "status": "ok"},
    )

    with pytest.raises(IrisSessionError):
        store.append_tool_event(
            "session_1",
            {"event_id": "tool_result:run_1:call_1", "status": "error"},
        )


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"event_id": "  "},
        {"event_id": 1},
        {"event_id": "tool_result:run_1:call_1", "not_json": {"set"}},
    ],
)
def test_memory_store_rejects_invalid_idempotent_events(event: dict[str, object]) -> None:
    store = InMemorySessionStore()

    with pytest.raises(IrisSessionError):
        store.append_tool_event("session_1", event)


def test_memory_store_rejects_inconsistent_existing_events() -> None:
    store = InMemorySessionStore()
    store._tool_events["session_1"] = [
        {"event_id": "tool_result:run_1:call_1", "status": "ok"},
        {"event_id": "tool_result:run_1:call_1", "status": "error"},
    ]

    with pytest.raises(IrisSessionError):
        store.append_tool_event(
            "session_1",
            {"event_id": "tool_result:run_1:call_1", "status": "ok"},
        )
