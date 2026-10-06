"""离线观测示例通过公开 Runner 产生实际调用树和恢复事实。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from opentelemetry.trace import StatusCode

from examples.observability.basic import run_example as run_basic
from examples.observability.streaming_child import run_example as run_streaming_child
from iris.harness import LiveFact
from iris.lifecycle import RunPhase, RunStopReason
from iris.message import ModelBlockDelta, ModelResponseCompleted, ModelResponseFailed
from iris.observability.service import Observability
from iris.runtime import RuntimeStreamEvent

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.asyncio
async def test_basic_example_exports_real_model_tool_model_tree(
    tmp_path: Path, observability: tuple[Observability, InMemorySpanExporter]
) -> None:
    observation, exporter = observability
    result = await run_basic(tmp_path, observability=observation)
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert "模型和工具" in result.assistant_message.text
    spans = exporter.get_finished_spans()
    [activation] = [span for span in spans if span.name.startswith("invoke_agent ")]
    models = [span for span in spans if span.attributes.get("gen_ai.operation.name") == "chat"]
    [tool] = [
        span for span in spans if span.attributes.get("gen_ai.operation.name") == "execute_tool"
    ]
    assert len(models) == 2
    assert activation.parent is None
    assert all(span.parent == activation.context for span in [*models, tool])
    assert [span.attributes["iris.step.index"] for span in models] == [0, 1]
    assert tool.attributes["iris.step.index"] == 0
    assert models[0].end_time <= tool.start_time <= tool.end_time <= models[1].start_time
    assert models[0].attributes["gen_ai.usage.input_tokens"] == 0
    assert "gen_ai.usage.output_tokens" not in models[0].attributes
    assert models[1].attributes["gen_ai.usage.output_tokens"] == 6
    assert "gen_ai.usage.input_tokens" not in models[1].attributes
    assert json.loads(tool.attributes["gen_ai.tool.call.arguments"])["file_path"] == "说明.txt"
    tool_output = json.loads(tool.attributes["gen_ai.tool.call.result"])
    assert "模型和工具" in json.dumps(tool_output, ensure_ascii=False)
    second_input = json.loads(models[1].attributes["gen_ai.input.messages"])
    [tool_part] = [
        part
        for message in second_input
        for part in message["parts"]
        if part["type"] == "tool_call_response"
    ]
    assert tool_part["id"] == "read-note"
    assert "模型和工具" in json.dumps(tool_part["response"], ensure_ascii=False)
    assert all(span.status.status_code is StatusCode.UNSET for span in spans)
    with observation.scope("借用者关闭后仍可记录"):
        pass
    assert exporter.get_finished_spans()[-1].name == "借用者关闭后仍可记录"


class RecordingPublisher:
    """保存公开 live plane 的 typed 事实，供示例行为断言。"""

    def __init__(self) -> None:
        """按实际交付顺序保存事实。"""
        self.facts: list[LiveFact] = []

    def publish(self, fact: LiveFact) -> None:
        """接收真实 runtime 和 lifecycle 事件。"""
        self.facts.append(fact)


@pytest.mark.asyncio
async def test_streaming_child_example_exports_wait_resume_control_and_failure(
    tmp_path: Path, observability: tuple[Observability, InMemorySpanExporter]
) -> None:
    observation, exporter = observability
    publisher = RecordingPublisher()
    waiting, completed, failed = await run_streaming_child(
        tmp_path, observability=observation, live_publisher=publisher
    )
    assert waiting.run.phase is RunPhase.WAITING
    assert waiting.pending_interaction.request.subagent_origin is not None
    assert completed.run.run_id == waiting.run.run_id
    assert completed.run.stop_reason is RunStopReason.COMPLETED
    assert failed.run.stop_reason is RunStopReason.FAILED
    assert "演示" in failed.error.message
    events = [
        fact.model_event
        for fact in publisher.facts
        if isinstance(fact, RuntimeStreamEvent) and fact.kind == "model.event"
    ]
    assert sum(isinstance(event, ModelResponseCompleted) for event in events) == 2
    assert sum(isinstance(event, ModelResponseFailed) for event in events) == 1
    assert any(isinstance(event, ModelBlockDelta) and event.channel == "text" for event in events)

    spans = exporter.get_finished_spans()
    models = [span for span in spans if span.attributes.get("gen_ai.operation.name") == "chat"]
    assert len(models) == 5
    [failed_model] = [span for span in models if span.status.status_code is StatusCode.ERROR]
    assert failed_model.attributes["iris.model.outcome"] == "failed"
    assert failed_model.attributes["gen_ai.usage.output_tokens"] == 0
    assert "gen_ai.usage.input_tokens" not in failed_model.attributes
    assert all(
        span.attributes["iris.model.outcome"] == "completed"
        for span in models
        if span is not failed_model
    )
    child_id = waiting.pending_interaction.request.subagent_origin.child_run_id
    drivers = [span for span in spans if span.name.startswith("invoke_agent ")]
    child_drivers = [span for span in drivers if span.attributes["iris.run.id"] == child_id]
    parent_drivers = [
        span for span in drivers if span.attributes["iris.run.id"] == waiting.run.run_id
    ]
    assert len(child_drivers) == len(parent_drivers) == 2
    assert (
        child_drivers[0].attributes["iris.activation.id"]
        != child_drivers[1].attributes["iris.activation.id"]
    )
    [control] = [
        span
        for span in spans
        if span.attributes.get("iris.control.operation") == "subagent_continue"
    ]
    assert child_drivers[1].parent == control.context
    assert control.end_time <= parent_drivers[1].start_time
    assert parent_drivers[1].parent is None
    [delegate] = [
        span
        for span in spans
        if span.attributes.get("gen_ai.tool.name") == "subagent"
        and span.attributes.get("gen_ai.operation.name") == "execute_tool"
    ]
    assert delegate.status.status_code is StatusCode.UNSET
    assert child_drivers[0].parent == delegate.context
    assert "gen_ai.tool.call.result" not in delegate.attributes
    assert not any(
        span.attributes.get("gen_ai.tool.name") == "ask_question"
        and span.attributes.get("gen_ai.operation.name") == "execute_tool"
        for span in spans
    )


@pytest.mark.asyncio
async def test_disabled_example_preserves_the_public_business_path(tmp_path: Path) -> None:
    result = await run_basic(tmp_path, observability=Observability())
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert (tmp_path / "说明.txt").exists()
