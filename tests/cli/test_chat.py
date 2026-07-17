from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import iris.cli.chat as chat_module
from iris.agents import AgentConfig
from iris.cli.chat import ChatOptions, run_chat_loop
from iris.cli.render import ChatRenderer
from iris.cli.trace import ChatTraceStore, TracingRuntimeProvider
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.hitl import (
    HumanInteraction,
    InteractionKind,
    InteractionStatus,
    PermissionInteractionRequest,
    PermissionInteractionResponse,
    QuestionInteractionRequest,
    QuestionInteractionResponse,
)
from iris.hitl.tools import AskQuestionTool
from iris.message import LLMRequest, LLMResponse, Msg, TextBlock, ToolUseBlock
from iris.runtime import AgentRuntime
from iris.runtime.models import RuntimeErrorInfo, RuntimeStatus, RuntimeTurnResult
from iris.session import InMemorySessionStore
from iris.tools import DefaultPermissionPolicy, ToolCapability, ToolExecutor, ToolRegistry


class FakeProvider:
    """按顺序返回响应并由 trace wrapper 记录请求。"""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """返回下一条响应。"""
        self.requests.append(request)
        return self.responses.pop(0)


class SequenceRuntime:
    """按顺序返回 waiting/resume 结果并记录调用上下文。"""

    def __init__(
        self,
        initial: RuntimeTurnResult,
        resumed: list[RuntimeTurnResult],
        *,
        resumable: HumanInteraction | None = None,
    ) -> None:
        self.initial = initial
        self.resumed = resumed
        self.resumable = resumable
        self.run_inputs: list[str] = []
        self.resume_calls: list[tuple[str, object | None]] = []
        self.loop_ids: list[int] = []
        self.recovery_sessions: list[str] = []

    def load_resumable_interaction(self, session_id: str) -> HumanInteraction | None:
        self.recovery_sessions.append(session_id)
        return self.resumable

    async def run_loop(
        self,
        user_input: str,
        *,
        options: Any = None,
    ) -> RuntimeTurnResult:
        del options
        self.run_inputs.append(user_input)
        self.loop_ids.append(id(asyncio.get_running_loop()))
        return self.initial

    async def resume(
        self,
        interaction_id: str,
        response: object | None = None,
    ) -> RuntimeTurnResult:
        self.resume_calls.append((interaction_id, response))
        self.loop_ids.append(id(asyncio.get_running_loop()))
        return self.resumed.pop(0)


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="cli-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是 CLI 助手。",
    )


def _context_input() -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(slots=[ContextSlot(name="instructions", content="保持简洁")])
    )


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        model="gpt-4o-mini",
        content=[TextBlock(text=text)],
        finish_reason="stop",
        input_tokens=2,
        output_tokens=3,
        total_tokens=5,
    )


def _runtime(trace_store: ChatTraceStore) -> AgentRuntime:
    provider = TracingRuntimeProvider(
        FakeProvider([_response("第一答复"), _response("第二答复")]),
        trace_store,
    )
    return AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=InMemorySessionStore(),
        workspace_root=Path.cwd(),
    )


def _renderer() -> tuple[ChatRenderer, Console]:
    console = Console(record=True, width=120, color_system=None)
    return ChatRenderer(console), console


def _permission_interaction() -> HumanInteraction:
    request = PermissionInteractionRequest(
        tool_call_id="call_write",
        tool_name="write_file",
        arguments={"path": "notes.md"},
        reason="工具需要写入工作区",
        workspace_root=str(Path.cwd()),
        call_fingerprint="a" * 64,
    )
    return HumanInteraction(
        interaction_id="int_11111111111111111111111111111111",
        session_id="demo",
        run_id="run-1",
        step_index=0,
        tool_call_id=request.tool_call_id,
        kind=InteractionKind.PERMISSION,
        request=request,
        checkpoint={},
    )


def _question_interaction() -> HumanInteraction:
    request = QuestionInteractionRequest(
        tool_call_id="call_question",
        question="请选择部署环境",
        options=["测试", "生产"],
    )
    return HumanInteraction(
        interaction_id="int_22222222222222222222222222222222",
        session_id="demo",
        run_id="run-1",
        step_index=0,
        tool_call_id=request.tool_call_id,
        kind=InteractionKind.QUESTION,
        request=request,
        checkpoint={},
    )


def _waiting_result(interaction: HumanInteraction) -> RuntimeTurnResult:
    tool_name = (
        interaction.request.tool_name
        if isinstance(interaction.request, PermissionInteractionRequest)
        else "ask_question"
    )
    return RuntimeTurnResult(
        session_id=interaction.session_id,
        run_id=interaction.run_id,
        status=RuntimeStatus.WAITING_HUMAN,
        assistant_message=Msg.assistant(
            [ToolUseBlock(id=interaction.tool_call_id, name=tool_name, input={})]
        ),
        steps=1,
        pending_interaction=interaction,
    )


def _terminal_result(text: str = "完成") -> RuntimeTurnResult:
    return RuntimeTurnResult(
        session_id="demo",
        run_id="run-1",
        status=RuntimeStatus.OK,
        assistant_message=Msg.assistant(text),
        steps=2,
    )


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("y", "approve"),
        ("yes", "approve"),
        ("", "reject"),
        ("n", "reject"),
        ("no", "reject"),
    ],
)
def test_collect_permission_response_maps_supported_tokens(
    token: str,
    expected: str,
) -> None:
    renderer, _ = _renderer()

    response = chat_module._collect_interaction_response(
        _permission_interaction(),
        input_func=lambda prompt: token,
        renderer=renderer,
    )

    assert isinstance(response, PermissionInteractionResponse)
    assert response.decision == expected


def test_collect_permission_response_reprompts_invalid_token() -> None:
    renderer, console = _renderer()
    inputs = iter(["later", "yes"])
    prompts: list[str] = []

    def read_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(inputs)

    response = chat_module._collect_interaction_response(
        _permission_interaction(),
        input_func=read_input,
        renderer=renderer,
    )

    assert isinstance(response, PermissionInteractionResponse)
    assert response.decision == "approve"
    assert len(prompts) == 2
    output = console.export_text()
    assert "y/yes/n/no" in output
    assert output.count(_permission_interaction().interaction_id) == 1


@pytest.mark.parametrize(("token", "expected"), [("1", "测试"), ("2", "生产")])
def test_collect_question_response_maps_numbered_option(
    token: str,
    expected: str,
) -> None:
    renderer, _ = _renderer()

    response = chat_module._collect_interaction_response(
        _question_interaction(),
        input_func=lambda prompt: token,
        renderer=renderer,
    )

    assert isinstance(response, QuestionInteractionResponse)
    assert response.answer == expected


def test_collect_question_response_reprompts_out_of_range_number() -> None:
    renderer, console = _renderer()
    inputs = iter(["3", "2"])

    response = chat_module._collect_interaction_response(
        _question_interaction(),
        input_func=lambda prompt: next(inputs),
        renderer=renderer,
    )

    assert isinstance(response, QuestionInteractionResponse)
    assert response.answer == "生产"
    output = console.export_text()
    assert "选项编号" in output
    assert output.count(_question_interaction().interaction_id) == 1


def test_collect_question_response_accepts_trimmed_free_text() -> None:
    renderer, _ = _renderer()

    response = chat_module._collect_interaction_response(
        _question_interaction(),
        input_func=lambda prompt: "  灰度环境  ",
        renderer=renderer,
    )

    assert isinstance(response, QuestionInteractionResponse)
    assert response.answer == "灰度环境"


def test_collect_question_response_reprompts_blank_answer() -> None:
    renderer, console = _renderer()
    inputs = iter(["   ", "测试"])

    response = chat_module._collect_interaction_response(
        _question_interaction(),
        input_func=lambda prompt: next(inputs),
        renderer=renderer,
    )

    assert isinstance(response, QuestionInteractionResponse)
    assert response.answer == "测试"
    output = console.export_text()
    assert "回答不能为空" in output
    assert output.count(_question_interaction().interaction_id) == 1


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, EOFError])
def test_collect_interaction_response_propagates_terminal_interrupt(
    error_type: type[BaseException],
) -> None:
    renderer, _ = _renderer()

    def raise_terminal_error(prompt: str) -> str:
        del prompt
        raise error_type

    with pytest.raises(error_type):
        chat_module._collect_interaction_response(
            _permission_interaction(),
            input_func=raise_terminal_error,
            renderer=renderer,
        )


def test_chat_loop_resumes_multiple_gates_in_order_on_one_event_loop() -> None:
    permission = _permission_interaction()
    question = _question_interaction()
    runtime = SequenceRuntime(
        _waiting_result(permission),
        [_waiting_result(question), _terminal_result("多 gate 完成")],
    )
    renderer, console = _renderer()
    inputs = iter(["开始", "y", "1", "/exit"])

    code = run_chat_loop(
        runtime=runtime,  # type: ignore[arg-type]
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=ChatTraceStore(),
        renderer=renderer,
        input_func=lambda prompt: next(inputs),
    )

    assert code == 0
    assert runtime.run_inputs == ["开始"]
    assert [interaction_id for interaction_id, _ in runtime.resume_calls] == [
        permission.interaction_id,
        question.interaction_id,
    ]
    assert isinstance(runtime.resume_calls[0][1], PermissionInteractionResponse)
    assert isinstance(runtime.resume_calls[1][1], QuestionInteractionResponse)
    assert len(set(runtime.loop_ids)) == 1
    output = console.export_text()
    assert "多 gate 完成" in output
    assert output.count("ASSISTANT") == 1


def test_chat_loop_invalid_interaction_input_does_not_resume_early() -> None:
    permission = _permission_interaction()
    runtime = SequenceRuntime(
        _waiting_result(permission),
        [_terminal_result()],
    )
    renderer, console = _renderer()
    inputs = iter(["开始", "later", "yes", "/exit"])

    code = run_chat_loop(
        runtime=runtime,  # type: ignore[arg-type]
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=ChatTraceStore(),
        renderer=renderer,
        input_func=lambda prompt: next(inputs),
    )

    assert code == 0
    assert len(runtime.resume_calls) == 1
    assert "y/yes/n/no" in console.export_text()


@pytest.mark.parametrize(
    ("error_type", "expected_code"),
    [(KeyboardInterrupt, 130), (EOFError, 0)],
)
def test_chat_loop_interaction_interrupt_preserves_pending(
    error_type: type[BaseException],
    expected_code: int,
) -> None:
    permission = _permission_interaction()
    runtime = SequenceRuntime(_waiting_result(permission), [])
    renderer, console = _renderer()
    input_calls = 0

    def read_input(prompt: str) -> str:
        nonlocal input_calls
        del prompt
        input_calls += 1
        if input_calls == 1:
            return "开始"
        raise error_type

    code = run_chat_loop(
        runtime=runtime,  # type: ignore[arg-type]
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=ChatTraceStore(),
        renderer=renderer,
        input_func=read_input,
    )

    assert code == expected_code
    assert runtime.resume_calls == []
    output = console.export_text()
    assert permission.interaction_id in output
    assert "ASSISTANT" not in output
    if error_type is KeyboardInterrupt:
        assert "interaction 保持 pending" in output


def test_chat_loop_resume_error_exits_without_reading_next_turn() -> None:
    permission = _permission_interaction()
    runtime = SequenceRuntime(
        _waiting_result(permission),
        [
            RuntimeTurnResult(
                session_id="demo",
                run_id="run-1",
                status=RuntimeStatus.ERROR,
                steps=1,
                error=RuntimeErrorInfo(
                    code="RESUME_FAILED",
                    message="恢复失败",
                    source="runtime",
                ),
            )
        ],
    )
    renderer, console = _renderer()
    inputs = iter(["开始", "y", "/exit"])
    input_calls = 0

    def read_input(prompt: str) -> str:
        nonlocal input_calls
        del prompt
        input_calls += 1
        return next(inputs)

    code = run_chat_loop(
        runtime=runtime,  # type: ignore[arg-type]
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=ChatTraceStore(),
        renderer=renderer,
        input_func=read_input,
    )

    assert code == 1
    assert input_calls == 2
    assert len(runtime.resume_calls) == 1
    assert "RESUME_FAILED" in console.export_text()


def test_chat_loop_real_runtime_handles_permission_question_and_trace_once() -> None:
    writes: list[str] = []

    def write_probe() -> str:
        writes.append("write")
        return "written"

    registry = ToolRegistry()
    registry.register_function(
        write_probe,
        description="写入探针",
        capabilities={ToolCapability.WRITE},
    )
    registry.register(AskQuestionTool())
    trace_store = ChatTraceStore()
    provider = FakeProvider(
        [
            LLMResponse(
                provider="fake",
                content=[
                    TextBlock(text="需要人工处理"),
                    ToolUseBlock(id="write", name="write_probe", input={}),
                    ToolUseBlock(
                        id="question",
                        name="ask_question",
                        input={"question": "请选择部署环境", "options": ["测试", "生产"]},
                    ),
                ],
                finish_reason="tool_calls",
            ),
            _response("全部完成"),
        ]
    )
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=TracingRuntimeProvider(provider, trace_store),
        session_store=InMemorySessionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(
            registry,
            permission_policy=DefaultPermissionPolicy(write_mode="confirm"),
        ),
        workspace_root=Path.cwd(),
    )
    renderer, console = _renderer()
    inputs = iter(["开始", "y", "1", "/exit"])

    code = run_chat_loop(
        runtime=runtime,
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=trace_store,
        renderer=renderer,
        input_func=lambda prompt: next(inputs),
    )

    assert code == 0
    assert writes == ["write"]
    assert len(provider.requests) == 2
    answer_contents = [
        block.content for message in provider.requests[1].messages for block in message.tool_results
    ]
    assert "测试" in answer_contents
    assert len(trace_store.steps_for_turn(1)) == 2
    output = console.export_text()
    assert "PERMISSION" in output
    assert "QUESTION" in output
    assert "全部完成" in output
    assert output.count("REQUEST 1.1") == 1
    assert output.count("REQUEST 1.2") == 1
    assert output.count("ASSISTANT") == 1


def test_chat_loop_real_runtime_rejects_permission_without_side_effect() -> None:
    writes: list[str] = []

    def write_probe() -> str:
        writes.append("write")
        return "written"

    registry = ToolRegistry()
    registry.register_function(
        write_probe,
        description="写入探针",
        capabilities={ToolCapability.WRITE},
    )
    trace_store = ChatTraceStore()
    provider = FakeProvider(
        [
            LLMResponse(
                provider="fake",
                content=[ToolUseBlock(id="write", name="write_probe", input={})],
                finish_reason="tool_calls",
            ),
            _response("拒绝后继续"),
        ]
    )
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=TracingRuntimeProvider(provider, trace_store),
        session_store=InMemorySessionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(
            registry,
            permission_policy=DefaultPermissionPolicy(write_mode="confirm"),
        ),
        workspace_root=Path.cwd(),
    )
    renderer, console = _renderer()
    inputs = iter(["开始", "", "/exit"])

    code = run_chat_loop(
        runtime=runtime,
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=trace_store,
        renderer=renderer,
        input_func=lambda prompt: next(inputs),
    )

    assert code == 0
    assert writes == []
    assert len(provider.requests) == 2
    assert "USER_REJECTED" in console.export_text()


def test_chat_loop_recovers_resolved_interaction_before_first_user_input() -> None:
    resolved = _question_interaction().model_copy(
        update={
            "status": InteractionStatus.RESOLVED,
            "response": QuestionInteractionResponse(answer="测试"),
        }
    )
    runtime = SequenceRuntime(
        _terminal_result("未调用"),
        [_terminal_result("恢复完成")],
        resumable=resolved,
    )
    renderer, console = _renderer()
    inputs = iter(["/exit"])

    code = run_chat_loop(
        runtime=runtime,  # type: ignore[arg-type]
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=ChatTraceStore(),
        renderer=renderer,
        input_func=lambda prompt: next(inputs),
    )

    assert code == 0
    assert runtime.recovery_sessions == ["demo"]
    assert runtime.run_inputs == []
    assert runtime.resume_calls == [(resolved.interaction_id, None)]
    output = console.export_text()
    assert "RECOVERY" in output
    assert "QUESTION" not in output
    assert "恢复完成" in output


def test_chat_loop_startup_recovery_error_exits_without_user_input() -> None:
    claimed = _question_interaction().model_copy(
        update={
            "status": InteractionStatus.CONSUMED,
            "response": QuestionInteractionResponse(answer="测试"),
        }
    )
    runtime = SequenceRuntime(
        _terminal_result("未调用"),
        [
            RuntimeTurnResult(
                session_id="demo",
                run_id="run-1",
                status=RuntimeStatus.ERROR,
                error=RuntimeErrorInfo(
                    code="HITL_EXECUTION_OUTCOME_UNKNOWN",
                    message="工具执行结果未知",
                    source="runtime",
                ),
            )
        ],
        resumable=claimed,
    )
    renderer, console = _renderer()

    def fail_on_input(prompt: str) -> str:
        raise AssertionError(f"不应读取普通输入: {prompt}")

    code = run_chat_loop(
        runtime=runtime,  # type: ignore[arg-type]
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=ChatTraceStore(),
        renderer=renderer,
        input_func=fail_on_input,
    )

    assert code == 1
    assert runtime.run_inputs == []
    assert runtime.resume_calls == [(claimed.interaction_id, None)]
    assert "HITL_EXECUTION_OUTCOME_UNKNOWN" in console.export_text()


def test_chat_loop_reuses_session_and_renders_trace() -> None:
    trace_store = ChatTraceStore()
    renderer, console = _renderer()
    inputs = iter(["第一轮", "第二轮", "/exit"])

    code = run_chat_loop(
        runtime=_runtime(trace_store),
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=trace_store,
        renderer=renderer,
        input_func=lambda prompt: next(inputs),
    )

    assert code == 0
    first_steps = trace_store.steps_for_turn(1)
    second_steps = trace_store.steps_for_turn(2)
    assert len(first_steps) == 1
    assert len(second_steps) == 1
    assert first_steps[0].request.messages[-1].text == "第一轮"
    assert any(message.text == "第一轮" for message in second_steps[0].request.messages)
    assert any(message.text == "第一答复" for message in second_steps[0].request.messages)

    output = console.export_text()
    assert "USER #1" in output
    assert "REQUEST 1.1" in output
    assert "RESPONSE 2.1" in output
    assert "第一答复" in output
    assert "第二答复" in output


def test_chat_loop_handles_slash_commands_without_provider_calls() -> None:
    trace_store = ChatTraceStore()
    renderer, console = _renderer()
    inputs = iter(["/help", "/trace full", "/exit"])

    code = run_chat_loop(
        runtime=_runtime(trace_store),
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=trace_store,
        renderer=renderer,
        input_func=lambda prompt: next(inputs),
    )

    assert code == 0
    assert trace_store.steps_for_turn(1) == []
    output = console.export_text()
    assert "HELP" in output
    assert "trace 已切换为 full" in output


def test_chat_loop_reuses_event_loop_across_turns() -> None:
    class LoopRecordingRuntime:
        """记录每轮调用所在的 event loop。"""

        def __init__(self) -> None:
            self.loop_ids: list[int] = []

        def load_resumable_interaction(self, session_id: str) -> None:
            del session_id
            return None

        async def run_loop(
            self,
            user_input: str,
            *,
            options: Any = None,
        ) -> RuntimeTurnResult:
            """返回空结果并记录当前 event loop。"""
            del user_input, options
            self.loop_ids.append(id(asyncio.get_running_loop()))
            return RuntimeTurnResult(
                session_id="demo",
                run_id="run-1",
                status=RuntimeStatus.OK,
                steps=1,
            )

    trace_store = ChatTraceStore()
    renderer, _ = _renderer()
    runtime = LoopRecordingRuntime()
    inputs = iter(["第一轮", "第二轮", "/exit"])

    code = run_chat_loop(
        runtime=runtime,  # type: ignore[arg-type]
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=trace_store,
        renderer=renderer,
        input_func=lambda prompt: next(inputs),
    )

    assert code == 0
    assert len(runtime.loop_ids) == 2
    assert runtime.loop_ids[0] == runtime.loop_ids[1]


def test_chat_options_validate_trace_mode() -> None:
    with pytest.raises(ValueError, match="trace_mode"):
        ChatOptions(config_path=Path("agent.yaml"), trace_mode="verbose")  # type: ignore[arg-type]
