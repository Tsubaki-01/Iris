"""摘要调用的分批重试、时间额度和取消行为。"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeRuntimeCommitPort, MutableCancellationSignal, build_runtime, start_activation

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.exceptions import IrisAPIConnectionError, IrisAuthenticationError
from iris.lifecycle import RuntimeExecutionOptions
from iris.message import LLMRequest, LLMResponse, Msg, TextBlock, ToolUseBlock
from iris.runtime import (
    AgentRuntime,
    RuntimeActivationInput,
    RuntimeActivationOutcome,
    RuntimeStreamEvent,
)


def _response(text: str = "压缩结果", *, finish_reason: str = "stop") -> LLMResponse:
    return LLMResponse(
        provider="fake",
        model="test",
        content=[TextBlock(text=text)],
        finish_reason=finish_reason,
        input_tokens=10,
        output_tokens=3,
        total_tokens=13,
    )


class _Provider:
    def __init__(
        self,
        actions: list[LLMResponse | Exception],
        *,
        on_summary: Callable[[], None] | None = None,
        delay: float = 0,
    ) -> None:
        self.actions = actions
        self.on_summary = on_summary
        self.delay = delay
        self.summaries: list[LLMRequest] = []
        self.main: list[LLMRequest] = []

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        if request.messages[0].text.startswith("You create a concise"):
            return len(request.messages[-1].text) + 500
        return 1000 if any("<summary>" in message.text for message in request.messages) else 9500

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if not request.messages[0].text.startswith("You create a concise"):
            self.main.append(request)
            return _response("完成")
        self.summaries.append(request)
        if self.on_summary is not None:
            self.on_summary()
        if self.delay:
            await asyncio.sleep(self.delay)
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _case(
    tmp_path: Path,
    provider: _Provider,
    *,
    history_chars: int = 100,
    operation_timeout: float = 300,
    request_timeout: float | None = None,
    prompt: Path | None = None,
) -> tuple[AgentRuntime, RuntimeActivationInput, FakeRuntimeCommitPort, MutableCancellationSignal]:
    activation = start_activation(
        initial_session_message_count=1,
        options=RuntimeExecutionOptions(request_options={"timeout": request_timeout}),
    )
    commits = FakeRuntimeCommitPort(activation, messages=[Msg.assistant("x" * history_chars)])
    runtime = build_runtime(
        agent_config=AgentConfig(
            name="compaction-controls",
            model={"provider": "openai", "name": "test"},
            system="test",
            compaction={
                "input_budget_tokens": 10000,
                "timeout_seconds": operation_timeout,
                "prompt": prompt,
            },
        ),
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="system", content="test")])
        ),
        provider=provider,
        workspace_root=tmp_path,
    )
    return runtime, activation, commits, MutableCancellationSignal()


@pytest.mark.asyncio
async def test_only_failed_batch_is_retried(tmp_path: Path) -> None:
    provider = _Provider([_response("A"), IrisAPIConnectionError("瞬断"), _response("B")])
    runtime, activation, commits, cancellation = _case(tmp_path, provider, history_chars=16000)
    result = await runtime.execute(activation, commits=commits, cancellation=cancellation)
    assert result.outcome == RuntimeActivationOutcome.COMPLETED
    assert len(provider.summaries) == 3
    assert provider.summaries[1].messages == provider.summaries[2].messages
    assert provider.summaries[0].messages != provider.summaries[1].messages
    assert all(request.provider_options["num_retries"] == 0 for request in provider.summaries)
    assert len(commits.compaction_usages) == 2
    assert len(commits.compaction_commits) == len(commits.model_commits) == 1


@pytest.mark.asyncio
async def test_summary_prompt_reloads_between_operations_not_batches(tmp_path: Path) -> None:
    prompt = tmp_path / "summary.j2"
    original_prompt = "You create a concise summary. Original instructions."
    updated_prompt = "You create a concise summary. Updated instructions."
    prompt.write_text(original_prompt, encoding="utf-8")
    original_mtime = prompt.stat().st_mtime

    def edit_prompt() -> None:
        prompt.write_text(updated_prompt, encoding="utf-8")
        os.utime(prompt, (original_mtime + 2, original_mtime + 2))
        provider.on_summary = None

    provider = _Provider(
        [
            _response("A"),
            IrisAPIConnectionError("瞬断"),
            _response("B"),
            _response("C"),
            _response("D"),
        ],
        on_summary=edit_prompt,
    )
    runtime, activation, commits, cancellation = _case(
        tmp_path, provider, history_chars=16000, prompt=prompt
    )

    result = await runtime.execute(activation, commits=commits, cancellation=cancellation)

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(provider.summaries) == 3
    assert all(request.messages[0].text == original_prompt for request in provider.summaries)
    assert provider.summaries[1].messages == provider.summaries[2].messages

    next_activation = start_activation(
        run_id="run-2", activation_id="activation-2", initial_session_message_count=1
    )
    next_commits = FakeRuntimeCommitPort(next_activation, messages=[Msg.assistant("y" * 16000)])
    next_result = await runtime.execute(
        next_activation, commits=next_commits, cancellation=MutableCancellationSignal()
    )

    assert next_result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(provider.summaries) == 5
    assert all(request.messages[0].text == updated_prompt for request in provider.summaries[3:])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type,attempts", [(IrisAuthenticationError, 1), (IrisAPIConnectionError, 2)]
)
async def test_terminal_provider_failure_keeps_previous_projection(
    tmp_path: Path, error_type: type[Exception], attempts: int
) -> None:
    provider = _Provider([_response("A"), error_type("失败"), error_type("失败")])
    runtime, activation, commits, cancellation = _case(tmp_path, provider, history_chars=16000)
    result = await runtime.execute(activation, commits=commits, cancellation=cancellation)
    assert result.outcome == RuntimeActivationOutcome.FAILED
    assert result.error.source == "provider"
    assert result.error.details["operation"] == "compaction"
    assert len(provider.summaries) == 1 + attempts
    assert len(commits.compaction_usages) == 1
    assert commits.compaction is None
    assert not provider.main


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        _response(""),
        _response("截断", finish_reason="length"),
        _response().model_copy(
            update={"content": [ToolUseBlock(id="unexpected", name="tool", input={})]}
        ),
    ],
)
async def test_invalid_summary_still_saves_usage(tmp_path: Path, response: LLMResponse) -> None:
    provider = _Provider([response])
    runtime, activation, commits, cancellation = _case(tmp_path, provider)
    result = await runtime.execute(activation, commits=commits, cancellation=cancellation)
    assert result.error.code == "CONTEXT_COMPACTION_FAILED"
    assert len(commits.compaction_usages) == 1
    assert len(provider.summaries) == 1
    assert not commits.compaction_commits


@pytest.mark.asyncio
async def test_operation_timeout_is_not_retried(tmp_path: Path) -> None:
    provider = _Provider([_response()], delay=0.1)
    runtime, activation, commits, cancellation = _case(tmp_path, provider, operation_timeout=0.01)
    result = await runtime.execute(activation, commits=commits, cancellation=cancellation)
    assert result.error.code == "CONTEXT_COMPACTION_TIMEOUT"
    assert len(provider.summaries) == 1
    assert 0 < provider.summaries[0].timeout <= 0.010001
    assert not commits.compaction_usages


@pytest.mark.asyncio
async def test_short_request_timeout_is_retried_once(tmp_path: Path) -> None:
    provider = _Provider([_response()], delay=0.1)
    runtime, activation, commits, cancellation = _case(tmp_path, provider, request_timeout=0.01)
    result = await runtime.execute(activation, commits=commits, cancellation=cancellation)
    assert result.error.source == "provider"
    assert result.error.code == "PROVIDER_TIMEOUT"
    assert len(provider.summaries) == 2
    assert all(request.timeout == 0.01 for request in provider.summaries)
    assert not commits.compaction_usages


@pytest.mark.asyncio
async def test_all_batches_share_one_operation_timeout(tmp_path: Path) -> None:
    provider = _Provider([_response("A"), _response("B")], delay=0.08)
    runtime, activation, commits, cancellation = _case(
        tmp_path, provider, history_chars=16000, operation_timeout=0.12
    )
    result = await runtime.execute(activation, commits=commits, cancellation=cancellation)
    assert result.error.code == "CONTEXT_COMPACTION_TIMEOUT"
    assert len(provider.summaries) == 2
    assert provider.summaries[1].timeout < provider.summaries[0].timeout
    assert len(commits.compaction_usages) == 1
    assert not commits.compaction_commits


@pytest.mark.asyncio
async def test_main_request_uses_deadline_remaining_after_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _Provider([_response()])
    runtime, activation, commits, cancellation = _case(tmp_path, provider)
    commits.deadline = 10
    provider.on_summary = lambda: setattr(commits, "deadline", 2)
    timeouts: list[float | None] = []
    original_wait = asyncio.wait_for

    async def capture_timeout(future: Coroutine[Any, Any, Any], timeout: float | None) -> Any:
        timeouts.append(timeout)
        return await original_wait(future, timeout)

    monkeypatch.setattr(asyncio, "wait_for", capture_timeout)
    result = await runtime.execute(activation, commits=commits, cancellation=cancellation)
    assert result.outcome == RuntimeActivationOutcome.COMPLETED
    assert timeouts == [10, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["cancel", "deadline"])
async def test_response_cannot_commit_projection_after_control_stop(
    tmp_path: Path, control: str
) -> None:
    provider = _Provider([_response()])
    runtime, activation, commits, cancellation = _case(tmp_path, provider)

    def stop() -> None:
        if control == "cancel":
            cancellation.requested = True
        else:
            commits.deadline = 0

    provider.on_summary = stop
    events: list[RuntimeStreamEvent] = []

    class Sink:
        def emit(self, event: RuntimeStreamEvent) -> None:
            events.append(event)

    result = await runtime.execute(
        activation, commits=commits, cancellation=cancellation, stream_sink=Sink()
    )
    expected = (
        RuntimeActivationOutcome.CANCELLED
        if control == "cancel"
        else RuntimeActivationOutcome.DEADLINE_EXCEEDED
    )
    assert result.outcome == expected
    assert len(commits.compaction_usages) == 1
    assert not commits.compaction_commits
    assert not provider.main
    assert [event.kind for event in events] == [
        "context.compaction.started",
        "context.compaction.failed",
    ]
