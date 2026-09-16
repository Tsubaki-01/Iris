"""自动压缩的完整请求额度、归档和主步骤契约。"""

from __future__ import annotations

from typing import Any

import pytest
from fakes import (
    FakeProvider,
    FakeRuntimeCommitPort,
    FakeRuntimeSteeringPort,
    MutableCancellationSignal,
    build_runtime,
    start_activation,
)

from iris.agents import AgentConfig
from iris.context import (
    ContextBuilder,
    ContextBuildInput,
    ContextBuildOutput,
    ContextSection,
    ContextSlot,
)
from iris.exceptions import IrisProviderError
from iris.lifecycle import RuntimeExecutionOptions, SessionCompaction
from iris.message import LLMRequest, LLMResponse, Msg, TextBlock
from iris.providers import ProviderClient
from iris.runtime import AgentRuntime, RuntimeActivationOutcome, RuntimeProvider, SteeringInput


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        finish_reason="stop",
        content=[TextBlock(text=text)],
        input_tokens=20,
        output_tokens=2,
        total_tokens=22,
    )


class _ThresholdProvider(FakeProvider):
    """分别控制完整原请求、规划请求、实际候选的估算值。"""

    def __init__(self, before: int, after: int) -> None:
        super().__init__([_response("新摘要"), _response("主回答")])
        self.before = before
        self.after = after
        self.estimates: list[LLMRequest] = []

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        self.estimates.append(request)
        if request.provider_options.get("num_retries") == 0:
            return 100
        if any(message.text == "<summary>\n\n</summary>" for message in request.messages):
            return 10
        if any(message.text == "<summary>\n新摘要\n</summary>" for message in request.messages):
            return self.after
        return self.before


def _runtime(provider: RuntimeProvider, *, builder: ContextBuilder | None = None) -> AgentRuntime:
    return build_runtime(
        agent_config=AgentConfig(name="compact", model="openai/gpt-4o-mini", system="规则"),
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="rules", content="业务规则")]),
            memory=ContextSection(slots=[ContextSlot(name="memory", content="记忆")]),
            before_current_input=ContextSection(
                slots=[ContextSlot(name="bci", content="当前环境")]
            ),
        ),
        provider=provider,
        context_builder=builder,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("before", "after", "accepted"),
    [
        (95000, 80000, False),
        (95000, 70000, True),
        (95000, 76800, True),
        (76800, 76800, False),
    ],
)
async def test_compaction_accepts_only_smaller_complete_request_within_trigger(
    before: int,
    after: int,
    accepted: bool,
) -> None:
    provider = _ThresholdProvider(before, after)
    raw = [Msg.user("旧任务"), Msg.assistant("旧结果")]
    activation = start_activation(input="当前任务", initial_session_message_count=len(raw))
    port = FakeRuntimeCommitPort(activation, messages=raw)
    result = await _runtime(provider).execute(
        activation,
        commits=port,
        cancellation=MutableCancellationSignal(),
    )

    assert len(port.compaction_usages) == 1
    assert port.compaction_usages[0].total_tokens == 22
    assert port.events.count("reserve_model_step") == 1
    assert len(provider.requests) == (2 if accepted else 1)
    assert len(port.compaction_commits) == int(accepted)
    if accepted:
        assert result.outcome == RuntimeActivationOutcome.COMPLETED
        assert provider.requests[-1].messages[-1].text == "当前任务"
        assert port.compaction_commits[0].after_input_tokens == after
        # 主响应归档只包含本turn输入和assistant；summary不进入原文。
        assert [message.text for message in port.messages[:2]] == ["旧任务", "旧结果"]
        assert all("<summary>" not in message.text for message in port.messages)
        assert [message.text for message in port.messages].count("当前任务") == 1
    else:
        assert result.error.code == "CONTEXT_COMPACTION_FAILED"
        assert port.messages == raw
        assert port.compaction is None
        assert port.model_commits == []


@pytest.mark.asyncio
async def test_below_trigger_does_not_compact_available_history() -> None:
    provider = _ThresholdProvider(76799, 1)
    provider._responses = [_response("主回答")]
    activation = start_activation(initial_session_message_count=1)
    port = FakeRuntimeCommitPort(activation, messages=[Msg.user("已有历史")])
    result = await _runtime(provider).execute(
        activation,
        commits=port,
        cancellation=MutableCancellationSignal(),
    )
    assert result.outcome == RuntimeActivationOutcome.COMPLETED
    assert port.compaction_commits == []
    assert port.compaction_usages == []
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_actual_main_provider_overflow_does_not_add_compaction_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ThresholdProvider(76799, 1)
    calls: list[LLMRequest] = []

    async def fail(request: LLMRequest) -> LLMResponse:
        calls.append(request)
        raise IrisProviderError("context window exceeded", status_code=400)

    monkeypatch.setattr(provider, "complete", fail)
    activation = start_activation(initial_session_message_count=1)
    port = FakeRuntimeCommitPort(activation, messages=[Msg.user("已有历史")])
    result = await _runtime(provider).execute(
        activation,
        commits=port,
        cancellation=MutableCancellationSignal(),
    )
    assert result.error.source == "provider"
    assert len(calls) == 1
    assert port.compaction_usages == []
    assert port.compaction_commits == []


@pytest.mark.asyncio
async def test_steer_queued_during_summary_waits_for_main_response_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ThresholdProvider(95000, 70000)
    provider._responses.append(_response("按新方向完成"))
    activation = start_activation(initial_session_message_count=1)
    port = FakeRuntimeCommitPort(activation, messages=[Msg.user("旧任务")])
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="new-direction", message=Msg.user("新方向"))]
    )
    complete = provider.complete

    async def observe(request: LLMRequest) -> LLMResponse:
        if request.provider_options.get("num_retries") == 0:
            assert steering.events == []
            assert all(message.text != "新方向" for message in port.messages)
            assert "新方向" not in request.messages[-1].text
        return await complete(request)

    monkeypatch.setattr(provider, "complete", observe)
    result = await _runtime(provider).execute(
        activation,
        commits=port,
        cancellation=MutableCancellationSignal(),
        steering=steering,
    )
    assert result.outcome == RuntimeActivationOutcome.COMPLETED
    assert len(provider.requests) == 3
    assert [message.text for message in port.messages].count("新方向") == 1
    assert provider.requests[-1].messages[-1].text == "新方向"
    assert steering.events[1] == ("acknowledge", "new-direction", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("before", [76800, 96000, 96001])
async def test_no_new_prefix_uses_input_budget_without_summary_call(before: int) -> None:
    provider = _ThresholdProvider(before, 1)
    provider._responses = [_response("主回答")]
    activation = start_activation()
    port = FakeRuntimeCommitPort(activation)
    result = await _runtime(provider).execute(
        activation,
        commits=port,
        cancellation=MutableCancellationSignal(),
    )
    assert port.compaction_usages == []
    assert port.compaction_commits == []
    if before <= 96000:
        assert result.outcome == RuntimeActivationOutcome.COMPLETED
        assert len(provider.requests) == 1
    else:
        assert result.error.code == "CONTEXT_COMPACTION_UNAVAILABLE"
        assert provider.requests == []


@pytest.mark.asyncio
async def test_existing_fully_covered_history_is_not_summarized_again() -> None:
    provider = _ThresholdProvider(80000, 1)
    provider._responses = [_response("主回答")]
    raw = [Msg.user("旧任务")]
    activation = start_activation(initial_session_message_count=1)
    port = FakeRuntimeCommitPort(activation, messages=raw)
    port.compaction = SessionCompaction(summary="已有摘要", covered_message_count=1)
    result = await _runtime(provider).execute(
        activation,
        commits=port,
        cancellation=MutableCancellationSignal(),
    )
    assert result.outcome == RuntimeActivationOutcome.COMPLETED
    assert len(provider.requests) == 1
    assert any(
        message.text == "<summary>\n已有摘要\n</summary>"
        for message in provider.requests[0].messages
    )
    assert port.compaction_commits == []


@pytest.mark.asyncio
async def test_budget_denial_prevents_any_summary_generation() -> None:
    provider = _ThresholdProvider(95000, 10000)
    activation = start_activation(initial_session_message_count=1)
    port = FakeRuntimeCommitPort(activation, messages=[Msg.user("旧任务")], max_model_steps=0)
    result = await _runtime(provider).execute(
        activation,
        commits=port,
        cancellation=MutableCancellationSignal(),
    )
    assert result.outcome == RuntimeActivationOutcome.BUDGET_EXHAUSTED
    assert provider.requests == []
    assert port.compaction_usages == []


class _CountingBuilder(ContextBuilder):
    """记录一次before_model中的context构建次数。"""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def build(self, input_data: ContextBuildInput) -> ContextBuildOutput:
        self.count += 1
        return super().build(input_data)


@pytest.mark.asyncio
async def test_candidate_reuses_context_and_effective_main_options() -> None:
    provider = _ThresholdProvider(95000, 70000)
    builder = _CountingBuilder()
    options = RuntimeExecutionOptions(
        request_options={
            "model": "effective-model",
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "provider_options": {"reasoning_effort": "low"},
        }
    )
    activation = start_activation(initial_session_message_count=1, options=options)
    port = FakeRuntimeCommitPort(activation, messages=[Msg.user("旧任务")])
    result = await _runtime(provider, builder=builder).execute(
        activation,
        commits=port,
        cancellation=MutableCancellationSignal(),
    )
    assert result.outcome == RuntimeActivationOutcome.COMPLETED
    assert builder.count == 1
    summary, main = provider.requests
    assert summary.model == main.model == "effective-model"
    assert summary.temperature == main.temperature == 0.1
    assert summary.response_format is None
    assert summary.provider_options == {"reasoning_effort": "low", "num_retries": 0}
    assert main.response_format == {"type": "json_object"}
    assert main.provider_options == {"reasoning_effort": "low"}
    assert all(
        request.response_format == main.response_format
        for request in provider.estimates
        if request.provider_options.get("num_retries") != 0
    )


@pytest.mark.asyncio
async def test_runtime_retry_reaches_provider_client_as_explicit_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """结合真实kwargs映射，runtime仅重试失败摘要一次，主调用不加重试覆盖。"""
    import iris.providers.client as client_module

    calls: list[dict[str, Any]] = []

    class RateLimitError(Exception):
        status_code = 429

    async def completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            raise RateLimitError("retry this summary")
        return {
            "choices": [
                {
                    "message": {"content": "新摘要" if len(calls) == 2 else "主回答"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
        }

    def estimate(self: ProviderClient, request: LLMRequest) -> int:
        if request.provider_options.get("num_retries") == 0:
            return 100
        if any("<summary>" in message.text for message in request.messages):
            return 10000
        return 95000

    monkeypatch.setattr(client_module.litellm, "acompletion", completion)
    monkeypatch.setattr(ProviderClient, "estimate_input_tokens", estimate)
    provider = ProviderClient(provider="openai", api_key="test")
    activation = start_activation(initial_session_message_count=1)
    port = FakeRuntimeCommitPort(activation, messages=[Msg.user("旧任务")])
    result = await _runtime(provider).execute(
        activation,
        commits=port,
        cancellation=MutableCancellationSignal(),
    )
    assert result.outcome == RuntimeActivationOutcome.COMPLETED
    assert len(calls) == 3
    assert calls[0]["num_retries"] == calls[1]["num_retries"] == 0
    assert "num_retries" not in calls[2]
    assert len(port.compaction_usages) == 1
