"""完整请求计量随候选传递，装配与压缩不会重复计量同一候选。"""

from pathlib import Path
from typing import Any

import pytest
from fakes import FakeRuntimeCommitPort, MutableCancellationSignal, start_activation

from iris.agents import ContextPolicyConfig
from iris.context import ContextContribution, ContextSnapshot
from iris.context.source import render_context_snapshot
from iris.lifecycle import SessionContextWindow
from iris.memory import MemoryService, SQLiteMemoryStore
from iris.message import LLMRequest, Msg
from iris.runtime import RuntimeActivationOutcome
from iris.runtime._context_projection import project_context_request
from iris.runtime._request_measurement import measure_request
from tests.runtime.test_compaction_execution import _runtime, _ThresholdProvider
from tests.runtime.test_context_projection_execution import CountingProvider
from tests.runtime.test_context_source import Source
from tests.runtime.test_context_source import _runtime as _source_runtime


@pytest.mark.parametrize("trigger", [1, 1000])
def test_unchanged_projection_preserves_request_and_its_measurement(trigger: int) -> None:
    """低压力和没有可撤材料的高压力投影都直接传递已有pair。"""
    provider = CountingProvider()
    request = LLMRequest(model="test", messages=[Msg.user("required")])
    measured = measure_request(request, provider.estimate_input_tokens)
    projected, _ = project_context_request(
        measured,
        source_indices={},
        config=ContextPolicyConfig(),
        trigger_tokens=trigger,
        estimate_input_tokens=provider.estimate_input_tokens,
    )
    assert projected is measured
    assert provider.estimates == [request]


def test_optional_changes_return_the_exact_last_measured_candidate() -> None:
    """动态材料和schema依序变化，每个新候选只估算一次。"""
    provider = CountingProvider()
    snapshot = ContextSnapshot((ContextContribution("optional", "notes", required=False),))
    request = LLMRequest(
        model="test",
        messages=[Msg.user("required"), render_context_snapshot(snapshot)],
        tools=[{"function": {"name": "extra"}}],
    )
    projected, selected = project_context_request(
        measure_request(request, provider.estimate_input_tokens),
        source_indices={},
        config=ContextPolicyConfig(),
        trigger_tokens=1,
        estimate_input_tokens=provider.estimate_input_tokens,
        snapshot=snapshot,
        select_optional=True,
        optional_tool_names=("extra",),
    )
    assert len(provider.estimates) == 3
    assert projected.request is provider.estimates[-1]
    assert projected.input_tokens == CountingProvider().estimate_input_tokens(projected.request)
    assert projected.request.tools == [] and selected.contributions == ()
    assert len(projected.request.messages) == 2
    assert request.tools and "notes" in request.messages[-1].text


@pytest.mark.asyncio
async def test_low_pressure_measures_the_complete_source_request_only_once() -> None:
    """source 先装配一次；未减载时压缩入口复用该请求的计量。"""
    provider, source = CountingProvider(), Source()
    runtime = _source_runtime(source, provider)
    activation = start_activation(input="question")
    result = await runtime.execute(
        activation,
        commits=FakeRuntimeCommitPort(activation),
        cancellation=MutableCancellationSignal(),
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(source.scopes) == 1
    assert len(provider.estimates) == 1
    assert provider.estimates[0].messages == provider.requests[0].messages
    assert (
        sum(
            message.metadata.get("context_kind") == "runtime_snapshot"
            for message in provider.estimates[0].messages
        )
        == 1
    )


@pytest.mark.asyncio
async def test_compaction_reuses_selected_memory_request_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """新概览请求已计量且无需正文减载时，直接用它验收并发送。"""
    import iris.runtime.runtime as runtime_module

    window = SessionContextWindow(memory_overview="new overview", mode="full")

    async def windows(**kwargs: Any) -> tuple[SessionContextWindow, SessionContextWindow]:
        return window, window

    monkeypatch.setattr(runtime_module, "load_context_windows", windows)
    provider = _ThresholdProvider(95000, 70000)
    runtime = _runtime(provider)
    runtime.environment.memory_service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"))
    raw = [Msg.user("old question"), Msg.assistant("old answer")]
    activation = start_activation(input="current", initial_session_message_count=len(raw))
    port = FakeRuntimeCommitPort(activation, messages=raw)
    port.context_window = SessionContextWindow(memory_overview="old overview", mode="full")
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(port.compaction_commits) == 1
    final = provider.requests[-1]
    matching = [request for request in provider.estimates if request.messages == final.messages]
    assert len(matching) == 1
    assert final.messages[0].text.endswith(window.memory_overview)
