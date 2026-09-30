"""一次 Todo 结束自查的模型请求、提交、预算与控制语义。"""

from pathlib import Path

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
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.exceptions import IrisRunPersistenceError
from iris.lifecycle import (
    CheckpointResumability,
)
from iris.message import (
    LLMRequest,
    LLMResponse,
    Msg,
    ToolUseBlock,
)
from iris.providers.protocols import CompletionProvider
from iris.runtime import (
    AgentRuntime,
    RuntimeActivationOutcome,
    RuntimeCursor,
    RuntimeModelStepCommit,
    SteeringInput,
)
from iris.tools import AskQuestionTool, DefaultPermissionPolicy, ToolRegistry, register_file_tools
from tests.runtime.test_execute import _text_response, _tool_response
from tests.runtime.test_todo_context import _write_todo


def _runtime(
    *,
    provider: CompletionProvider,
    tmp_path: Path,
    registry: ToolRegistry | None = None,
    enabled: bool = True,
) -> AgentRuntime:
    return build_runtime(
        agent_config=AgentConfig(
            name="todo-reminder",
            model="openai/test",
            system="维护工作清单",
            todo={"enabled": enabled},
            permissions={"workspace": str(tmp_path), "writes": "allow"},
            compaction={"input_budget_tokens": 10000},
        ),
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="rules", content="维护工作清单")])
        ),
        provider=provider,
        tool_registry=registry,
        permission_policy=DefaultPermissionPolicy(write_mode="allow"),
        workspace_root=tmp_path,
    )


@pytest.mark.asyncio
async def test_unfinished_todo_commits_candidate_then_reminds_once(tmp_path: Path) -> None:
    """未完成项首次结束时触发下一步，第二次仍可正常结束且不伪造用户历史。"""
    _write_todo(tmp_path, "session-1", "- [ ] awaiting-user-work")
    provider = FakeProvider([_text_response("candidate"), _text_response("still waiting")])
    runtime = _runtime(provider=provider, tmp_path=tmp_path)
    activation = start_activation(input="检查任务")
    port = FakeRuntimeCommitPort(activation, max_model_steps=4)
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(provider.requests) == 2
    assert "Todo 结束自查" not in provider.requests[0].messages[-1].text
    assert "Todo 结束自查" in provider.requests[1].messages[-1].text
    first, final = port.model_commits
    assert first.message_delta == (first.assistant_message,)
    assert first.cursor_after.position == "before_model"
    assert first.cursor_after.step_index == first.cursor_after.todo_reminder_step == 1
    assert first.resumability is CheckpointResumability.SAFE
    assert final.cursor_after.position == "outcome_ready"
    assert final.cursor_after.todo_reminder_step == 1
    assert result.assistant_message.text == "still waiting"
    assert [message.text for message in port.messages] == ["检查任务", "candidate", "still waiting"]
    assert sum(commit.total_tokens for commit in port.model_commits) == 12


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, "", "# 清单", "- [x] finished", "- [?] invalid"])
async def test_empty_completed_and_diagnostic_todo_trigger_rules(
    tmp_path: Path, content: str | None
) -> None:
    """缺失、空与全完成不提醒，格式诊断需要且只需要一次自查。"""
    if content is not None:
        _write_todo(tmp_path, "session-1", content)
    invalid = content == "- [?] invalid"
    provider = FakeProvider([_text_response("candidate"), _text_response("diagnostic explained")])
    activation = start_activation()
    port = FakeRuntimeCommitPort(activation, max_model_steps=3)
    result = await _runtime(provider=provider, tmp_path=tmp_path).execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(provider.requests) == (2 if invalid else 1)
    assert result.cursor.todo_reminder_step == (1 if invalid else None)
    if invalid:
        assert "Todo 结束自查" in provider.requests[1].messages[-1].text
        assert "格式错误" in provider.requests[1].messages[-1].text


@pytest.mark.asyncio
async def test_scheduled_target_rereads_human_cleared_file_and_still_checks(tmp_path: Path) -> None:
    """提醒计划提交后清空文件，目标步仍带提醒但不会补回旧条目。"""
    path = _write_todo(tmp_path, "session-1", "- [-] old-work-item")

    class ClearingPort(FakeRuntimeCommitPort):
        """在原子提醒转换完成后模拟人工清空文件。"""

        def commit_model_step(self, commit: RuntimeModelStepCommit) -> RuntimeCursor:
            """先完成真实 fake 转换校验，再改变外部清单。"""
            cursor = super().commit_model_step(commit)
            if cursor.position == "before_model":
                path.write_text("", encoding="utf-8")
            return cursor

    provider = FakeProvider([_text_response("candidate"), _text_response("cleared")])
    activation = start_activation()
    port = ClearingPort(activation, max_model_steps=3)
    result = await _runtime(provider=provider, tmp_path=tmp_path).execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(provider.requests) == 2
    current = provider.requests[1].messages[-1].text
    assert "Todo 结束自查" in current and "当前清单为空" in current
    assert "old-work-item" not in current
    assert result.cursor.todo_reminder_step == 1


@pytest.mark.asyncio
async def test_reminder_can_continue_through_real_file_tools_without_repeating(
    tmp_path: Path,
) -> None:
    """自查引出多步真实读写，控制标记及 read-state 保留，后续步骤不重复指令。"""
    path = _write_todo(tmp_path, "session-1", "- [ ] finish-now\n- [ ] waiting-for-user")
    provider = FakeProvider(
        [
            _text_response("candidate"),
            _tool_response(
                ToolUseBlock(id="read", name="read_file", input={"file_path": str(path)})
            ),
            _tool_response(
                ToolUseBlock(
                    id="write",
                    name="write_file",
                    input={
                        "file_path": str(path),
                        "content": "- [x] finish-now\n- [ ] waiting-for-user",
                    },
                )
            ),
            _text_response("waiting-for-user remains"),
        ]
    )
    activation = start_activation()
    port = FakeRuntimeCommitPort(activation, max_model_steps=6)
    result = await _runtime(
        provider=provider, tmp_path=tmp_path, registry=register_file_tools()
    ).execute(activation, commits=port, cancellation=MutableCancellationSignal())
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(provider.requests) == 4
    assert ["Todo 结束自查" in request.messages[-1].text for request in provider.requests] == [
        False,
        True,
        False,
        False,
    ]
    assert len(port.tool_commits) == 2
    assert all(not commit.result.is_error for commit in port.tool_commits)
    assert all(commit.cursor_after.todo_reminder_step == 1 for commit in port.model_commits)
    assert all(commit.cursor_after.todo_reminder_step == 1 for commit in port.tool_commits)
    assert port.tool_commits[0].cursor_after.read_state is not None
    assert "- [x] finish-now" in provider.requests[-1].messages[-1].text
    assert "- [ ] waiting-for-user" in path.read_text(encoding="utf-8")
    assert sum(commit.total_tokens for commit in port.model_commits) == 28
    assert port.events.count("reserve_model_step") == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_last_granted_step_accepts_reply_without_extra_reservation(
    tmp_path: Path, enabled: bool
) -> None:
    """最后获准模型步骤直接接受正常回复，开关均不改变预算结算。"""
    _write_todo(tmp_path, "session-1", "- [ ] unfinished")
    provider = FakeProvider([_text_response("last available reply")])
    activation = start_activation()
    port = FakeRuntimeCommitPort(activation, max_model_steps=1)
    result = await _runtime(provider=provider, tmp_path=tmp_path, enabled=enabled).execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert result.cursor.todo_reminder_step is None
    assert len(provider.requests) == port.events.count("reserve_model_step") == 1
    assert result.assistant_message.text == "last available reply"


@pytest.mark.asyncio
async def test_steer_has_priority_and_later_steer_preserves_reminder_marker(tmp_path: Path) -> None:
    """真实输入先于提醒，提醒后的新输入只推进步骤而不重新开放自查。"""
    _write_todo(tmp_path, "session-1", "- [ ] still-open")
    first_steer = SteeringInput(submission_id="first", message=Msg.user("first steer"))
    later_steer = SteeringInput(submission_id="later", message=Msg.user("later steer"))
    steering = FakeRuntimeSteeringPort([first_steer])

    class SteeringProvider(FakeProvider):
        """在已安排的提醒步骤完成前送入另一次真实输入。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """按请求顺序注入第二条 steer。"""
            response = await super().complete(request)
            if len(self.requests) == 3:
                steering.inputs.append(later_steer)
            return response

    provider = SteeringProvider([_text_response(f"reply-{step}") for step in range(4)])
    activation = start_activation()
    port = FakeRuntimeCommitPort(activation, max_model_steps=6)
    result = await _runtime(provider=provider, tmp_path=tmp_path).execute(
        activation, commits=port, cancellation=MutableCancellationSignal(), steering=steering
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert [commit.cursor_after.todo_reminder_step for commit in port.model_commits] == [
        None,
        2,
        2,
        2,
    ]
    assert [len(commit.message_delta) for commit in port.model_commits] == [2, 1, 2, 1]
    assert ["Todo 结束自查" in request.messages[-1].text for request in provider.requests] == [
        False,
        False,
        True,
        False,
    ]
    assert [message.text for message in port.messages if message.role.value == "user"] == [
        "当前问题",
        "first steer",
        "later steer",
    ]
    assert [event[1] for event in steering.events if event[0] == "acknowledge"] == [
        "first",
        "later",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["deadline", "cancel"])
async def test_response_time_control_cannot_revive_a_todo_step(
    tmp_path: Path, control: str
) -> None:
    """使用响应后的最新 deadline；取消仍取消，均不能追加自查步骤。"""
    _write_todo(tmp_path, "session-1", "- [ ] unfinished")
    signal = MutableCancellationSignal()

    class ControlProvider(FakeProvider):
        """模拟 provider 返回前刚刚到期或收到取消。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """先记录有效响应，然后改变本 Run 控制状态。"""
            response = await super().complete(request)
            if control == "deadline":
                port.deadline = 0
            else:
                signal.requested = True
            return response

    provider = ControlProvider([_text_response("candidate")])
    activation = start_activation()
    port = FakeRuntimeCommitPort(activation, max_model_steps=3, remaining_deadline_seconds=100)
    result = await _runtime(provider=provider, tmp_path=tmp_path).execute(
        activation, commits=port, cancellation=signal
    )
    assert result.outcome is (
        RuntimeActivationOutcome.COMPLETED
        if control == "deadline"
        else RuntimeActivationOutcome.CANCELLED
    )
    assert result.cursor.todo_reminder_step is None
    assert len(provider.requests) == port.events.count("reserve_model_step") == 1
    assert len(port.model_commits) == (1 if control == "deadline" else 0)


@pytest.mark.asyncio
async def test_question_suspension_does_not_start_todo_reminder(tmp_path: Path) -> None:
    """需要用户回答时保持 WAITING 对应的 suspension，不为 Todo 继续调用模型。"""
    _write_todo(tmp_path, "session-1", "- [ ] awaiting-answer")
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    provider = FakeProvider(
        [_tool_response(ToolUseBlock(id="ask", name="ask_question", input={"question": "继续？"}))]
    )
    activation = start_activation()
    port = FakeRuntimeCommitPort(activation, max_model_steps=3)
    result = await _runtime(provider=provider, tmp_path=tmp_path, registry=registry).execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.SUSPENDED
    assert result.cursor.todo_reminder_step is None
    assert len(provider.requests) == 1
    assert len(port.suspensions) == 1


@pytest.mark.asyncio
async def test_failed_reminder_commit_does_not_publish_control_state(tmp_path: Path) -> None:
    """候选响应提交失败沿既有持久化异常退出，不提前改变 cursor 或请求下一步。"""
    _write_todo(tmp_path, "session-1", "- [ ] pending")
    provider = FakeProvider([_text_response("candidate")])
    activation = start_activation()
    port = FakeRuntimeCommitPort(activation, max_model_steps=3, fail_at="commit_model_step")
    with pytest.raises(IrisRunPersistenceError, match="commit"):
        await _runtime(provider=provider, tmp_path=tmp_path).execute(
            activation, commits=port, cancellation=MutableCancellationSignal()
        )
    assert port.cursor.position == "before_model" and port.cursor.todo_reminder_step is None
    assert len(provider.requests) == 1 and port.model_commits == []
    assert [message.role.value for message in port.messages] == ["user"]
