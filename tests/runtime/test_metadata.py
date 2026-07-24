from __future__ import annotations

from iris.exceptions import IrisSessionError
from iris.message import LLMResponse, TextBlock
from iris.runtime.metadata import (
    build_resume_run_metadata,
    build_run_metadata,
    synchronize_resume_metadata,
)
from iris.runtime.models import RuntimeStatus, RuntimeTurnResult
from iris.session import InMemorySessionStore


class _FailingMetadataStore(InMemorySessionStore):
    def save_run_metadata(self, session_id: str, metadata: dict[str, object]) -> None:
        del session_id, metadata
        raise IrisSessionError("metadata write failed")


def test_build_run_metadata_appends_complete_snapshot() -> None:
    response = LLMResponse(
        provider="fake",
        model="gpt-4o-mini",
        content=[TextBlock(text="done")],
        finish_reason="stop",
        input_tokens=2,
        output_tokens=3,
        total_tokens=5,
    )

    metadata = build_run_metadata(
        existing={"runs": [{"run_id": "old"}]},
        session_id="session-1",
        run_id="run-1",
        status=RuntimeStatus.OK,
        provider="openai",
        response=response,
        message_count=2,
        metadata={"trace_id": "trace-1"},
        tool_count=1,
    )

    latest = metadata["latest_run"]
    assert isinstance(latest, dict)
    assert latest["run_id"] == "run-1"
    assert latest["model"] == "gpt-4o-mini"
    assert latest["total_tokens"] == 5
    assert latest["tool_count"] == 1
    assert metadata["runs"] == [{"run_id": "old"}, latest]


def test_build_resume_run_metadata_replaces_markers_and_appends_snapshot() -> None:
    result = RuntimeTurnResult(
        session_id="session-1",
        run_id="run-1",
        status=RuntimeStatus.OK,
        steps=2,
    )

    metadata = build_resume_run_metadata(
        existing={
            "latest_run": {
                "run_id": "run-1",
                "waiting_human": True,
                "interaction_id": "int_old",
                "error": {"code": "OLD"},
                "keep": "value",
            },
            "runs": [],
        },
        result=result,
        message_count=4,
    )

    latest = metadata["latest_run"]
    assert isinstance(latest, dict)
    assert latest["keep"] == "value"
    assert latest["status"] == "ok"
    assert latest["message_count"] == 4
    assert "waiting_human" not in latest
    assert "interaction_id" not in latest
    assert "error" not in latest
    assert metadata["runs"] == [latest]


def test_synchronize_resume_metadata_writes_snapshot_or_returns_session_error() -> None:
    result = RuntimeTurnResult(
        session_id="session-1",
        run_id="run-1",
        status=RuntimeStatus.OK,
    )
    store = InMemorySessionStore()
    store.save_messages("session-1", [{"role": "user", "content": "hello"}])

    synchronized = synchronize_resume_metadata(session_store=store, result=result)
    failed = synchronize_resume_metadata(
        session_store=_FailingMetadataStore(),
        result=result,
    )

    assert synchronized is result
    assert store.load_run_metadata("session-1")["latest_run"]["message_count"] == 1
    assert failed.status is RuntimeStatus.ERROR
    assert failed.error is not None
    assert failed.error.source == "session"
