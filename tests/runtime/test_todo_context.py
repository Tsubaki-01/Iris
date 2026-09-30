"""Todo 文件在获准模型步骤中采样，并随动态上下文进入真实请求。"""

import asyncio
from pathlib import Path

import pytest
from fakes import (
    FakeRuntimeCommitPort,
    FakeRuntimeSteeringPort,
    MutableCancellationSignal,
    build_runtime,
    start_activation,
)

import iris.runtime.runtime as runtime_module
from iris.agents import AgentConfig
from iris.context import (
    ContextBuildInput,
    ContextBuildScope,
    ContextContribution,
    ContextSection,
    ContextSlot,
    ContextSnapshot,
)
from iris.exceptions import IrisAPIConnectionError, IrisCancellationRequestedError, IrisTodoError
from iris.lifecycle import RuntimeExecutionOptions
from iris.message import LLMRequest, LLMResponse, Msg
from iris.runtime import AgentRuntime, RuntimeActivationOutcome, RuntimeCursor, SteeringInput
from iris.todo import TodoSnapshot
from iris.todo.document import read_todo
from tests.runtime.test_context_projection_execution import CountingProvider


def _write_todo(workspace: Path, session_id: str, content: str) -> Path:
    path = workspace / ".iris" / "todos" / f"{session_id.encode('utf-8').hex()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _runtime(workspace: Path, provider: CountingProvider, *, enabled: bool = True) -> AgentRuntime:
    return build_runtime(
        agent_config=AgentConfig(
            name="todo-context",
            model="openai/test",
            system="stable",
            todo={"enabled": enabled},
            compaction={"input_budget_tokens": 10000},
        ),
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="rules", content="stable")])
        ),
        provider=provider,
        workspace_root=workspace,
    )


@pytest.mark.asyncio
async def test_todo_without_host_source_enters_request_but_not_history(tmp_path: Path) -> None:
    """没有宿主 source 也投影当前清单，原始历史保持用户与 assistant 消息。"""
    path = _write_todo(tmp_path, "session-1", "- [ ] current-todo-evidence")
    provider = CountingProvider()
    runtime = _runtime(tmp_path, provider)
    activation = start_activation(options=RuntimeExecutionOptions(include_tools=False))
    port = FakeRuntimeCommitPort(activation, max_model_steps=1)
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(provider.requests) == 1
    context = provider.requests[0].messages[-1]
    assert context.metadata["context_kind"] == "runtime_snapshot"
    assert "[iris.todo]" in context.text
    assert str(path) in context.text
    assert "- [ ] current-todo-evidence" in context.text
    assert "read_file" in context.text and "write_file" in context.text
    assert all("current-todo-evidence" not in message.text for message in port.messages)


@pytest.mark.asyncio
async def test_missing_todo_still_enters_request_without_creating_file(tmp_path: Path) -> None:
    """空工作区仍向模型给出当前文件位置和维护指令，读取不创建文件。"""
    provider = CountingProvider()
    runtime = _runtime(tmp_path, provider)
    activation = start_activation(options=RuntimeExecutionOptions(include_tools=False))
    result = await runtime.execute(
        activation,
        commits=FakeRuntimeCommitPort(activation, max_model_steps=1),
        cancellation=MutableCancellationSignal(),
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    context = provider.requests[0].messages[-1].text
    expected = tmp_path / ".iris/todos" / f"{b'session-1'.hex()}.md"
    assert str(expected) in context and "当前清单为空" in context
    assert "read_file" in context and "write_file" in context
    assert not expected.parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["disabled", "budget", "cancel", "deadline"])
async def test_disabled_or_unadmitted_step_does_not_read_todo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    """关闭功能或尚未获准的模型步骤不能采样文件。"""
    path = _write_todo(tmp_path, "session-1", "- [ ] must-not-read")
    reads: list[str] = []

    async def tracked_read(workspace: Path, session_id: str) -> TodoSnapshot:
        reads.append(session_id)
        return await read_todo(workspace, session_id)

    monkeypatch.setattr(runtime_module, "read_todo", tracked_read)
    provider = CountingProvider()
    runtime = _runtime(tmp_path, provider, enabled=control != "disabled")
    activation = start_activation(options=RuntimeExecutionOptions(include_tools=False))
    port = FakeRuntimeCommitPort(activation, max_model_steps=0 if control == "budget" else 1)
    if control == "deadline":
        port.deadline = 0
    result = await runtime.execute(
        activation,
        commits=port,
        cancellation=MutableCancellationSignal(requested=control == "cancel"),
    )
    assert (
        result.outcome.value
        == {
            "disabled": "completed",
            "budget": "budget_exhausted",
            "cancel": "cancelled",
            "deadline": "deadline_exceeded",
        }[control]
    )
    assert reads == []
    assert len(provider.requests) == (1 if control == "disabled" else 0)
    assert all(
        "[iris.todo]" not in message.text and str(path) not in message.text
        for request in provider.requests
        for message in request.messages
    )


@pytest.mark.asyncio
async def test_human_edit_is_visible_on_next_admitted_model_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """一次运行的相邻模型步骤各读一次，人工替换不会保留旧动态清单。"""
    path = _write_todo(tmp_path, "session-1", "- [ ] todo-before-edit")
    reads: list[TodoSnapshot] = []

    async def tracked_read(workspace: Path, session_id: str) -> TodoSnapshot:
        assert port.events.count("reserve_model_step") == len(reads) + 1
        snapshot = await read_todo(workspace, session_id)
        reads.append(snapshot)
        return snapshot

    class EditingProvider(CountingProvider):
        """第一步请求进入 provider 后模拟用户保存清单。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """捕获本步请求，再让下一步骤读取人工修改后的状态。"""
            response = await super().complete(request)
            if len(self.requests) == 1:
                path.write_text("- [x] todo-after-edit", encoding="utf-8")
            return response

    monkeypatch.setattr(runtime_module, "read_todo", tracked_read)
    provider = EditingProvider()
    runtime = _runtime(tmp_path, provider)
    activation = start_activation(options=RuntimeExecutionOptions(include_tools=False))
    port = FakeRuntimeCommitPort(activation, max_model_steps=2)
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="continue", message=Msg.user("继续检查"))]
    )
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal(), steering=steering
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(reads) == len(provider.requests) == 2
    first, second = (request.messages[-1].text for request in provider.requests)
    assert "todo-before-edit" in first and "todo-after-edit" not in first
    assert "todo-after-edit" in second and "todo-before-edit" not in second
    assert all("todo-before-edit" not in message.text for message in port.messages)
    assert all("todo-after-edit" not in message.text for message in port.messages)


@pytest.mark.asyncio
async def test_concurrent_sessions_keep_request_local_todo_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一个 runtime 的并发读取完成顺序不同，也不能串用其他 session 清单。"""
    sessions = ("first-session", "second-session")
    paths = {
        session: _write_todo(tmp_path, session, f"- [x] {session}-todo") for session in sessions
    }
    barrier = asyncio.Barrier(2)
    reads: list[str] = []

    async def overlapping_read(workspace: Path, session_id: str) -> TodoSnapshot:
        snapshot = await read_todo(workspace, session_id)
        reads.append(session_id)
        await barrier.wait()
        return snapshot

    monkeypatch.setattr(runtime_module, "read_todo", overlapping_read)
    provider = CountingProvider()
    runtime = _runtime(tmp_path, provider)
    activations = [
        start_activation(
            input=f"input-{session}",
            session_id=session,
            run_id=f"run-{session}",
            activation_id=f"activation-{session}",
            options=RuntimeExecutionOptions(include_tools=False),
        )
        for session in sessions
    ]
    results = await asyncio.gather(
        *(
            runtime.execute(
                activation,
                commits=FakeRuntimeCommitPort(activation, max_model_steps=1),
                cancellation=MutableCancellationSignal(),
            )
            for activation in activations
        )
    )
    assert all(result.outcome is RuntimeActivationOutcome.COMPLETED for result in results)
    assert sorted(reads) == sorted(sessions)
    assert len(provider.requests) == 2
    for request in provider.requests:
        original_input = request.messages[-2].text
        session = original_input.removeprefix("input-")
        context = request.messages[-1].text
        assert str(paths[session]) in context and f"{session}-todo" in context
        other = next(candidate for candidate in sessions if candidate != session)
        assert str(paths[other]) not in context and f"{other}-todo" not in context


@pytest.mark.asyncio
async def test_summary_retry_uses_one_frozen_todo_read_and_excludes_it_from_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """摘要重试中外部编辑文件，主请求仍使用本步采集值，摘要原料没有动态清单。"""
    path = _write_todo(tmp_path, "session-1", "- [ ] frozen-todo-evidence")
    reads: list[TodoSnapshot] = []

    async def tracked_read(workspace: Path, session_id: str) -> TodoSnapshot:
        snapshot = await read_todo(workspace, session_id)
        reads.append(snapshot)
        return snapshot

    class Source:
        """同时注入 required 与可减载的宿主材料。"""

        calls = 0

        async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
            """返回固定当前状态并记录实际采集次数。"""
            self.calls += 1
            return ContextSnapshot(
                (
                    ContextContribution("host-state", "host-current-only"),
                    ContextContribution("optional-host", "discard-me" * 600, required=False),
                )
            )

    class RetryProvider(CountingProvider):
        """首个摘要调用修改文件并报告一次可重试失败。"""

        retried = False

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """保留失败请求，随后沿普通摘要与主请求路径响应。"""
            if request.provider_options.get("num_retries") == 0 and not self.retried:
                self.retried = True
                self.requests.append(request)
                path.write_text("- [x] edited-during-summary", encoding="utf-8")
                raise IrisAPIConnectionError("one retry")
            return await super().complete(request)

    monkeypatch.setattr(runtime_module, "read_todo", tracked_read)
    provider, source = RetryProvider(), Source()
    runtime = _runtime(tmp_path, provider)
    runtime.environment.context_source = source
    raw = [Msg.assistant("archived evidence" * 1000), Msg.user("original")]
    activation = start_activation(
        input="original",
        initial_session_message_count=1,
        options=RuntimeExecutionOptions(include_tools=False),
    ).model_copy(
        update={
            "kind": "recover",
            "cursor": RuntimeCursor(
                todo_reminder_step=None,
                position="before_model",
                step_index=0,
                visible_tool_names=(),
            ),
        }
    )
    port = FakeRuntimeCommitPort(activation, messages=raw, max_model_steps=1)
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert source.calls == len(reads) == len(port.compaction_commits) == 1
    summaries = [
        request for request in provider.requests if request.provider_options.get("num_retries") == 0
    ]
    assert len(summaries) >= 2 and summaries[0].messages == summaries[1].messages
    assert all(
        "frozen-todo-evidence" not in message.text
        and "edited-during-summary" not in message.text
        and "host-current-only" not in message.text
        for request in summaries
        for message in request.messages
    )
    final = provider.requests[-1]
    context = final.messages[-1].text
    assert "frozen-todo-evidence" in context and "host-current-only" in context
    assert "edited-during-summary" not in context and "discard-me" not in context
    assert (
        sum(
            message.metadata.get("context_kind") == "runtime_snapshot" for message in final.messages
        )
        == 1
    )
    assert (
        provider.estimate_input_tokens(final)
        <= runtime.environment.agent_config.compaction.trigger_tokens
    )
    assert port.messages[:2] == raw
    assert "edited-during-summary" in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_oversized_required_todo_is_not_removed_to_fit_request(tmp_path: Path) -> None:
    """Todo 超过硬输入预算时正常报告压缩失败，不能丢掉清单继续调用。"""
    _write_todo(tmp_path, "session-1", "- [ ] " + "required-todo" * 1000)
    provider = CountingProvider()
    runtime = _runtime(tmp_path, provider)
    activation = start_activation(options=RuntimeExecutionOptions(include_tools=False))
    result = await runtime.execute(
        activation,
        commits=FakeRuntimeCommitPort(activation, max_model_steps=1),
        cancellation=MutableCancellationSignal(),
    )
    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.error is not None
    assert result.error.code == "CONTEXT_COMPACTION_UNAVAILABLE"
    assert result.error.source == "context"
    assert provider.requests == []


@pytest.mark.asyncio
async def test_malformed_document_enters_request_as_diagnostic(tmp_path: Path) -> None:
    """Markdown 格式错误是模型可见的修复材料，不转成 activation 失败。"""
    path = _write_todo(tmp_path, "session-1", "# Todo\n- [ ] valid-prefix\n- [?] malformed")
    provider = CountingProvider()
    runtime = _runtime(tmp_path, provider)
    activation = start_activation(options=RuntimeExecutionOptions(include_tools=False))
    result = await runtime.execute(
        activation,
        commits=FakeRuntimeCommitPort(activation, max_model_steps=1),
        cancellation=MutableCancellationSignal(),
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    context = provider.requests[0].messages[-1].text
    assert str(path) in context and "第 3 行格式错误" in context
    assert "valid-prefix" not in context and "当前清单为空" not in context


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["read", "template"])
async def test_todo_domain_failure_retains_runtime_code_and_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """真实 I/O 与模板失败保留 Todo 领域码和路径，不被 host context 包装。"""
    from iris.todo import context as todo_context

    path = _write_todo(tmp_path, "session-1", "- [ ] example")
    if failure == "read":
        path.unlink()
        path.mkdir()
        detail_key = "path"
        failed_path = path
    else:
        failed_path = tmp_path / "broken_todo_context.j2"
        failed_path.write_text("{% invalid %}", encoding="utf-8")
        monkeypatch.setattr(todo_context, "_TEMPLATE", failed_path)
        detail_key = "template"
    provider = CountingProvider()
    runtime = _runtime(tmp_path, provider)
    activation = start_activation(options=RuntimeExecutionOptions(include_tools=False))
    result = await runtime.execute(
        activation,
        commits=FakeRuntimeCommitPort(activation, max_model_steps=1),
        cancellation=MutableCancellationSignal(),
    )
    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.error is not None
    assert result.error.code == "TODO_ERROR" and result.error.source == "runtime"
    assert result.error.details[detail_key] == str(failed_path)
    assert provider.requests == []


@pytest.mark.asyncio
async def test_reserved_todo_key_conflict_is_context_error(tmp_path: Path) -> None:
    """宿主抢占保留 key 时合并失败，不能悄悄覆盖任一贡献。"""
    _write_todo(tmp_path, "session-1", "- [ ] real-todo")

    class Source:
        """模拟宿主误用 framework 保留 key。"""

        async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
            """返回一个与 Todo 内置贡献冲突的宿主快照。"""
            return ContextSnapshot((ContextContribution("iris.todo", "host-todo"),))

    provider = CountingProvider()
    runtime = _runtime(tmp_path, provider)
    runtime.environment.context_source = Source()
    activation = start_activation(options=RuntimeExecutionOptions(include_tools=False))
    result = await runtime.execute(
        activation,
        commits=FakeRuntimeCommitPort(activation, max_model_steps=1),
        cancellation=MutableCancellationSignal(),
    )
    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.error is not None
    assert result.error.code == "CONTEXT_ERROR" and result.error.source == "context"
    assert "iris.todo" in result.error.message
    assert provider.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["signal", "task", "deadline"])
async def test_todo_read_preserves_cancellation_and_timeout_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    """采样期间取消保持控制语义，已触发的 deadline 优先于随后报告的 Todo 错误。"""
    _write_todo(tmp_path, "session-1", "- [ ] waiting-todo")
    entered = asyncio.Event()
    budgets: list[asyncio.Timeout] = []
    original_timeout = asyncio.timeout

    def tracked_timeout(delay: float | None) -> asyncio.Timeout:
        budget = original_timeout(delay)
        budgets.append(budget)
        return budget

    async def waiting_read(workspace: Path, session_id: str) -> TodoSnapshot:
        snapshot = await read_todo(workspace, session_id)
        entered.set()
        if control == "signal":
            raise IrisCancellationRequestedError("todo read cancelled")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if control == "deadline":
                raise IrisTodoError("reader interrupted after deadline") from None
            raise
        return snapshot

    monkeypatch.setattr(runtime_module, "read_todo", waiting_read)
    monkeypatch.setattr(asyncio, "timeout", tracked_timeout)
    provider = CountingProvider()
    runtime = _runtime(tmp_path, provider)
    activation = start_activation(options=RuntimeExecutionOptions(include_tools=False))
    port = FakeRuntimeCommitPort(activation, max_model_steps=1, remaining_deadline_seconds=100)
    task = asyncio.create_task(
        runtime.execute(activation, commits=port, cancellation=MutableCancellationSignal())
    )
    await entered.wait()
    if control == "task":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        if control == "deadline":
            budgets[-1].reschedule(asyncio.get_running_loop().time())
        result = await task
        assert result.outcome is (
            RuntimeActivationOutcome.CANCELLED
            if control == "signal"
            else RuntimeActivationOutcome.DEADLINE_EXCEEDED
        )
        assert result.error is None
    assert provider.requests == []
