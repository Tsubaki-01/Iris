"""压缩 durable 模型的加载边界约束。"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunPhase,
    RunRecord,
    RunStopReason,
    RunUsage,
    SessionCompaction,
    SessionSnapshot,
    TokenUsage,
)
from iris.message import Msg


def test_session_compaction_checks_raw_prefix_at_load_boundary() -> None:
    compact = SessionCompaction(summary="摘要", covered_message_count=2)
    snapshot = SessionSnapshot(
        session_id="s", messages=[Msg.user("a"), Msg.user("b")], compaction=compact
    )
    assert SessionSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot
    with pytest.raises(ValidationError, match="覆盖"):
        SessionSnapshot(session_id="s", messages=[Msg.user("a")], compaction=compact)
    with pytest.raises(ValidationError):
        SessionCompaction(summary=" \n ", covered_message_count=1)


def test_summary_usage_keeps_independent_provider_counters() -> None:
    usage = RunUsage(
        input_tokens=26000,
        total_tokens=26000,
        compaction=TokenUsage(input_tokens=20000, output_tokens=2000, total_tokens=22001),
    )
    loaded = RunUsage.model_validate_json(usage.model_dump_json())
    assert loaded.input_tokens == 26000
    assert loaded.compaction.total_tokens == 22001
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=-1)


def test_terminal_compaction_is_bounded_by_frozen_message_count() -> None:
    now = datetime.now(UTC)
    record = RunRecord(
        run_id="r",
        session_id="s",
        agent_id="a",
        request=AgentRunRequest(input="hello", session_id="s", run_id="r"),
        options=AgentRunOptions(),
        initial_session_message_count=1,
        phase=RunPhase.TERMINAL,
        stop_reason=RunStopReason.COMPLETED,
        terminal_session_message_count=2,
        terminal_compaction=SessionCompaction(summary="摘要", covered_message_count=2),
        revision=1,
        checkpoint_sequence=0,
        last_event_sequence=1,
        created_at=now,
        started_at=now,
        updated_at=now,
        finished_at=now,
    )
    raw = record.model_dump()
    assert RunRecord.model_validate(raw).terminal_compaction == record.terminal_compaction
    raw["terminal_session_message_count"] = 1
    with pytest.raises(ValidationError, match="覆盖"):
        RunRecord.model_validate(raw)
    raw["terminal_compaction"] = None
    assert RunRecord.model_validate(raw).terminal_compaction is None
    del raw["initial_session_message_count"]
    with pytest.raises(ValidationError, match="initial_session_message_count"):
        RunRecord.model_validate(raw)
