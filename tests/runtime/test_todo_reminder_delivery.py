"""Todo 自查在摘要重试、恢复目标步骤及真实流式交付中的行为。"""

from pathlib import Path

import pytest
from fakes import (
    FakeRuntimeCommitPort,
    FakeStreamingProvider,
    MutableCancellationSignal,
    start_activation,
)

import iris.runtime.runtime as runtime_module
from iris.agents import AgentConfig
from iris.exceptions import IrisAPIConnectionError
from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunEvent,
    RunEventKind,
    RunLimits,
    RunStopReason,
    RuntimeExecutionOptions,
)
from iris.message import LLMRequest, LLMResponse, ModelBlockDelta, ModelResponseCompleted, Msg
from iris.runtime import RuntimeActivationOutcome, RuntimeCursor, RuntimeStreamEvent
from iris.todo import TodoSnapshot
from iris.todo.document import read_todo
from tests.harness.fakes import RecordingPublisher
from tests.runtime.test_context_projection_execution import CountingProvider
from tests.runtime.test_streaming import _stream_events, _text_response
from tests.runtime.test_todo_context import _runtime, _write_todo


@pytest.mark.asyncio
@pytest.mark.parametrize("recover_target", [False, True])
async def test_target_compaction_retry_keeps_frozen_hint_out_of_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recover_target: bool
) -> None:
    """普通和恢复目标步都只采一次文件，摘要重试不能吞掉或归档自查指令。"""
    path = _write_todo(tmp_path, "session-1", "- [ ] todo-before-summary")
    reads: list[TodoSnapshot] = []

    async def tracked_read(workspace: Path, session_id: str) -> TodoSnapshot:
        snapshot = await read_todo(workspace, session_id)
        reads.append(snapshot)
        return snapshot

    class RetryProvider(CountingProvider):
        """候选长回复制造目标步压力，首次摘要调用可重试失败。"""

        retried = False

        def __init__(self) -> None:
            super().__init__()
            self.main_responses = iter(
                ["最终答复"] if recover_target else ["candidate-evidence" * 1000, "最终答复"]
            )

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """在摘要重试之间真实改写文件，主请求仍必须使用已冻结快照。"""
            self.requests.append(request)
            if request.provider_options.get("num_retries") == 0:
                if not self.retried:
                    self.retried = True
                    path.write_text("- [x] edited-during-reminder-summary", encoding="utf-8")
                    raise IrisAPIConnectionError("retry reminder summary")
                return _text_response("已归纳原始工作历史")
            return _text_response(next(self.main_responses))

    monkeypatch.setattr(runtime_module, "read_todo", tracked_read)
    provider = RetryProvider()
    runtime = _runtime(tmp_path, provider)
    activation = start_activation(
        input="original", options=RuntimeExecutionOptions(include_tools=False)
    )
    messages: list[Msg] = []
    if recover_target:
        messages = [
            Msg.assistant("archived-evidence" * 1000),
            Msg.user("original"),
            Msg.assistant("先前已提交的候选回复"),
        ]
        activation = activation.model_copy(
            update={
                "kind": "recover",
                "initial_session_message_count": 1,
                "cursor": RuntimeCursor(
                    position="before_model",
                    step_index=1,
                    todo_reminder_step=1,
                    visible_tool_names=(),
                ),
            }
        )
    port = FakeRuntimeCommitPort(activation, messages=messages, max_model_steps=2)
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert result.assistant_message.text == "最终答复"
    assert result.cursor.todo_reminder_step == 1
    assert len(reads) == (1 if recover_target else 2)
    assert len(port.compaction_commits) == 1
    summaries = [
        request for request in provider.requests if request.provider_options.get("num_retries") == 0
    ]
    assert provider.retried and len(summaries) >= 2
    assert summaries[0].messages == summaries[1].messages
    assert all(
        "Todo 结束自查" not in message.text
        and "todo-before-summary" not in message.text
        and "edited-during-reminder-summary" not in message.text
        for request in summaries
        for message in request.messages
    )
    final = provider.requests[-1]
    [snapshot] = [
        message
        for message in final.messages
        if message.metadata.get("context_kind") == "runtime_snapshot"
    ]
    assert snapshot.text.count("Todo 结束自查") == 1
    assert "todo-before-summary" in snapshot.text
    assert "edited-during-reminder-summary" not in snapshot.text
    assert (
        provider.estimate_input_tokens(final)
        <= runtime.environment.agent_config.compaction.trigger_tokens
    )
    assert "edited-during-reminder-summary" in path.read_text(encoding="utf-8")
    assert all("Todo 结束自查" not in message.text for message in port.messages)


@pytest.mark.asyncio
async def test_streamed_candidate_precedes_final_and_only_final_terminates_run(
    tmp_path: Path,
) -> None:
    """候选回复照常流式显示，真正末次回复成为结果且只发布一次 completed 终态。"""
    _write_todo(tmp_path, "work", "- [ ] 等待人工反馈")
    candidate = _text_response("候选回复")
    final = _text_response("自查后仍在等待反馈，本轮结束")
    provider = FakeStreamingProvider(
        [
            _stream_events(candidate, stream_id="candidate"),
            _stream_events(final, stream_id="final"),
        ]
    )
    publisher = RecordingPublisher()
    runner = AgentRunner.from_config(
        AgentConfig(
            name="todo-stream",
            model="openai/test",
            system="维护清单并如实解释状态。",
            permissions={"workspace": str(tmp_path)},
            todo={"enabled": True},
        ),
        provider=provider,
        live_publisher=publisher,
    )
    try:
        result = await runner.start(
            AgentRunRequest(input="处理任务", session_id="work", run_id="stream"),
            options=AgentRunOptions(limits=RunLimits(max_model_steps=4)),
        )
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert result.assistant_message.text == final.to_msg().text
        assert result.run.usage.model_steps_committed == 2
        assert len(provider.stream_requests) == 2
        assert provider.requests == []
        assert "Todo 结束自查" not in provider.stream_requests[0].messages[-1].text
        assert "Todo 结束自查" in provider.stream_requests[1].messages[-1].text
        deltas = [
            (fact.step_index, fact.model_event.delta)
            for fact in publisher.facts
            if isinstance(fact, RuntimeStreamEvent)
            and isinstance(fact.model_event, ModelBlockDelta)
        ]
        assert deltas == [(0, candidate.to_msg().text), (1, final.to_msg().text)]
        completions = [
            (index, fact.model_event.response.to_msg().text)
            for index, fact in enumerate(publisher.facts)
            if isinstance(fact, RuntimeStreamEvent)
            and isinstance(fact.model_event, ModelResponseCompleted)
        ]
        assert [text for _, text in completions] == [candidate.to_msg().text, final.to_msg().text]
        [(terminal_index, terminal)] = [
            (index, fact)
            for index, fact in enumerate(publisher.facts)
            if isinstance(fact, RunEvent) and fact.kind is RunEventKind.RUN_TERMINAL
        ]
        assert completions[0][0] < completions[1][0] < terminal_index
        assert terminal.payload["stop_reason"] == "completed"
        assert runner.get_result("stream") == result
        assert [message.text for message in runner.get_session("work").messages] == [
            "处理任务",
            candidate.to_msg().text,
            final.to_msg().text,
        ]
    finally:
        await runner.aclose()
