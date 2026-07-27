from __future__ import annotations

import pytest

from iris.exceptions import IrisSessionError, IrisToolExecutionError
from iris.message import Msg, TextBlock
from iris.runtime.tool_result_committer import (
    commit_tool_results,
    project_tool_result_messages,
)
from iris.session import InMemorySessionStore
from iris.tools import ToolErrorInfo, ToolResult


def _result(
    *,
    tool_use_id: str = "call-1",
    tool_name: str = "echo",
    text: str = "完成",
) -> ToolResult:
    return ToolResult(
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        content=[TextBlock(text=text)],
    )


def test_commit_tool_results_projects_and_persists_batch_in_order() -> None:
    store = InMemorySessionStore()
    store.save_messages("session-1", [Msg.user("问题").model_dump(mode="json")])
    results = [
        _result(tool_use_id="call-1", tool_name="first", text="一"),
        _result(tool_use_id="call-2", tool_name="second", text="二"),
    ]

    committed = commit_tool_results(
        results=results,
        session_store=store,
        session_id="session-1",
        run_id="run-1",
        step_index=2,
        agent_id="agent-1",
        metadata={"trace_id": "trace-1"},
        deduplicate_messages=False,
    )

    assert committed.results == results
    assert [message.tool_results[0].tool_use_id for message in committed.messages] == [
        "call-1",
        "call-2",
    ]
    assert [event["event_id"] for event in committed.events] == [
        "tool_result:run-1:call-1",
        "tool_result:run-1:call-2",
    ]
    assert [message["role"] for message in store.load_messages("session-1")] == [
        "user",
        "user",
        "user",
    ]
    assert store.load_tool_events("session-1") == committed.events


def test_project_tool_result_messages_preserves_result_order() -> None:
    messages = project_tool_result_messages(
        [
            _result(tool_use_id="call-1", text="一"),
            _result(tool_use_id="call-2", text="二"),
        ]
    )

    assert [message.tool_results[0].tool_use_id for message in messages] == [
        "call-1",
        "call-2",
    ]


def test_commit_tool_results_skips_store_for_empty_batch() -> None:
    store = RecordingSessionStore()

    committed = commit_tool_results(
        results=[],
        session_store=store,
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        agent_id="agent-1",
        metadata=None,
        deduplicate_messages=False,
    )

    assert committed.results == []
    assert committed.messages == []
    assert committed.events == []
    assert store.calls == []


def test_commit_tool_results_deduplicates_result_ready_message_and_event() -> None:
    store = InMemorySessionStore()
    result = _result()

    first = commit_tool_results(
        results=[result],
        session_store=store,
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        agent_id="agent-1",
        metadata=None,
        deduplicate_messages=True,
    )
    second = commit_tool_results(
        results=[result],
        session_store=store,
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        agent_id="agent-1",
        metadata=None,
        deduplicate_messages=True,
    )

    assert [message.tool_results[0].tool_use_id for message in first.messages] == ["call-1"]
    assert [message.tool_results[0].tool_use_id for message in second.messages] == ["call-1"]
    assert len(store.load_messages("session-1")) == 1
    assert len(store.load_tool_events("session-1")) == 1


def test_commit_tool_results_preserves_event_payload_conflict() -> None:
    store = InMemorySessionStore()
    result = _result()
    commit_tool_results(
        results=[result],
        session_store=store,
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        agent_id="agent-1",
        metadata=None,
        deduplicate_messages=True,
    )
    changed = result.model_copy(
        update={
            "is_error": True,
            "error": ToolErrorInfo(code="FAILED", message="失败"),
        }
    )

    with pytest.raises(IrisSessionError, match="payload 不一致"):
        commit_tool_results(
            results=[changed],
            session_store=store,
            session_id="session-1",
            run_id="run-1",
            step_index=0,
            agent_id="agent-1",
            metadata=None,
            deduplicate_messages=True,
        )

    assert len(store.load_messages("session-1")) == 1
    assert len(store.load_tool_events("session-1")) == 1


def test_commit_tool_results_rejects_non_json_metadata_before_store_write() -> None:
    store = RecordingSessionStore()

    with pytest.raises(IrisToolExecutionError, match="非 JSON 可序列化"):
        commit_tool_results(
            results=[_result()],
            session_store=store,
            session_id="session-1",
            run_id="run-1",
            step_index=0,
            agent_id="agent-1",
            metadata={"invalid": {"set"}},
            deduplicate_messages=False,
        )

    assert store.calls == []


class RecordingSessionStore(InMemorySessionStore):
    """记录 committer 访问的 session store 方法。"""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def load_messages(self, session_id: str) -> list[dict[str, object]]:
        """记录消息读取。"""
        self.calls.append("load_messages")
        return super().load_messages(session_id)

    def save_messages(self, session_id: str, messages: list[dict[str, object]]) -> None:
        """记录消息写入。"""
        self.calls.append("save_messages")
        super().save_messages(session_id, messages)

    def append_tool_event(self, session_id: str, event: dict[str, object]) -> None:
        """记录 event 写入。"""
        self.calls.append("append_tool_event")
        super().append_tool_event(session_id, event)
