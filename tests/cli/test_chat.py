from __future__ import annotations

import threading
from pathlib import Path

from iris.cli.chat import (
    ChatOptions,
    _parse_interaction_response,
    _write_interaction_prompt,
    run_chat_loop,
)
from iris.exceptions import IrisProviderError, IrisRunStateError
from iris.harness import AgentRunner
from iris.hitl import (
    HumanInteraction,
    HumanInteractionRequest,
    InteractionStatus,
    PermissionInteractionResponse,
    PermissionPrompt,
    QuestionInteractionResponse,
    QuestionPrompt,
    ToolCallSnapshot,
)
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry
from tests.harness.fakes import StaticProvider, build_runtime, text_response, tool_response


def _interaction(prompt: PermissionPrompt | QuestionPrompt) -> HumanInteraction:
    subject = ToolCallSnapshot(
        tool_call_id="call-1",
        tool_name="write_file",
        arguments={"path": "note.txt", "content": "hello"},
        workspace_root=str(Path.cwd()),
        fingerprint="a" * 64,
    )
    return HumanInteraction(
        interaction_id="interaction-1",
        session_id="cli",
        run_id="run-1",
        step_index=0,
        tool_call_id="call-1",
        status=InteractionStatus.PENDING,
        request=HumanInteractionRequest(tool_call=subject, prompt=prompt),
    )


class SequenceRunner:
    """仅供不启动 run 的终端命令测试使用。"""

    def __init__(self) -> None:
        self.start_calls: list[object] = []


class ErrorRunner:
    """在 manager create admission 前抛出领域错误。"""

    async def _start_managed(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise IrisRunStateError("状态不允许")


class FailingProvider:
    """始终返回 provider 领域错误。"""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        raise IrisProviderError("调用失败", provider="fake")


def test_chat_loop_resumes_permission_through_session_manager(tmp_path: Path) -> None:
    """CLI 展示 waiting prompt，并把下一行作为 exact typed resume。"""

    def write(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        name="write",
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="write-1", name="write", input={"value": "x"})),
        text_response("已完成"),
    )
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider, registry=registry),
        store=store,
    )
    prompted = threading.Event()
    finished = threading.Event()
    outputs: list[str] = []
    errors: list[str] = []
    input_index = 0

    def read_input(prompt: str) -> str:
        nonlocal input_index
        del prompt
        input_index += 1
        if input_index == 1:
            return "执行"
        if input_index == 2:
            assert prompted.wait(1)
            return "y"
        assert finished.wait(1)
        return "/exit"

    def write_output(message: str) -> None:
        outputs.append(message)
        if message == "批准该调用？ [y/N]":
            prompted.set()
        if message == "已完成":
            finished.set()

    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=read_input,
        output_func=write_output,
        error_func=errors.append,
    )

    assert code == 0
    assert len(provider.requests) == 2
    assert store.load_session("cli").messages[-1].text == "已完成"
    assert any("已完成" in line for line in outputs)
    assert errors == []


def test_chat_loop_maps_second_question_option(tmp_path: Path) -> None:
    del tmp_path
    interaction = _interaction(QuestionPrompt(question="选择格式", options=["Markdown", "纯文本"]))
    outputs: list[str] = []

    _write_interaction_prompt(interaction, output_func=outputs.append)
    response = _parse_interaction_response(
        interaction,
        "2",
        output_func=outputs.append,
    )

    assert response == QuestionInteractionResponse(answer="纯文本")
    assert any("1. Markdown" in line for line in outputs)
    assert any("2. 纯文本" in line for line in outputs)


def test_chat_loop_accepts_free_text_question_answer(tmp_path: Path) -> None:
    del tmp_path
    interaction = _interaction(QuestionPrompt(question="补充说明", options=["跳过", "继续"]))
    outputs: list[str] = []

    response = _parse_interaction_response(
        interaction,
        "自定义回答",
        output_func=outputs.append,
    )

    assert response == QuestionInteractionResponse(answer="自定义回答")
    assert outputs == []


def test_chat_loop_displays_permission_details(tmp_path: Path) -> None:
    del tmp_path
    interaction = _interaction(PermissionPrompt(reason="写入笔记"))
    outputs: list[str] = []

    _write_interaction_prompt(interaction, output_func=outputs.append)
    response = _parse_interaction_response(
        interaction,
        "n",
        output_func=outputs.append,
    )

    output = "\n".join(outputs)
    assert "write_file" in output
    assert "note.txt" in output
    assert "写入笔记" in output
    assert "本次批准只适用于该调用" in output
    assert response == PermissionInteractionResponse(decision="reject")


def test_chat_loop_help_lists_exit_and_quit_without_trace(tmp_path: Path) -> None:
    runner = SequenceRunner()
    answers = iter(["/help", "/exit"])
    outputs: list[str] = []
    errors: list[str] = []

    code = run_chat_loop(
        runner=runner,  # type: ignore[arg-type]
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=lambda prompt: next(answers),
        output_func=outputs.append,
        error_func=errors.append,
    )

    output = "\n".join(outputs)
    assert code == 0
    assert "/exit" in output
    assert "/quit" in output
    assert "/trace" not in output
    assert runner.start_calls == []
    assert errors == []


def test_chat_loop_quit_returns_zero_without_start(tmp_path: Path) -> None:
    runner = SequenceRunner()
    answers = iter(["/quit", "执行", "/exit"])
    outputs: list[str] = []
    errors: list[str] = []

    code = run_chat_loop(
        runner=runner,  # type: ignore[arg-type]
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=lambda prompt: next(answers),
        output_func=outputs.append,
        error_func=errors.append,
    )

    assert code == 0
    assert runner.start_calls == []
    assert errors == []


def test_chat_loop_returns_zero_on_eof(tmp_path: Path) -> None:
    runner = SequenceRunner()
    outputs: list[str] = []
    errors: list[str] = []

    def raise_eof(prompt: str) -> str:
        del prompt
        raise EOFError

    code = run_chat_loop(
        runner=runner,  # type: ignore[arg-type]
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=raise_eof,
        output_func=outputs.append,
        error_func=errors.append,
    )

    assert code == 0
    assert errors == []


def test_chat_loop_returns_130_on_keyboard_interrupt(tmp_path: Path) -> None:
    runner = SequenceRunner()
    outputs: list[str] = []
    errors: list[str] = []

    def raise_keyboard_interrupt(prompt: str) -> str:
        del prompt
        raise KeyboardInterrupt

    code = run_chat_loop(
        runner=runner,  # type: ignore[arg-type]
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=raise_keyboard_interrupt,
        output_func=outputs.append,
        error_func=errors.append,
    )

    assert code == 130
    assert errors == []


def test_chat_loop_formats_iris_error(tmp_path: Path) -> None:
    answers = iter(["执行"])
    outputs: list[str] = []
    errors: list[str] = []

    code = run_chat_loop(
        runner=ErrorRunner(),  # type: ignore[arg-type]
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=lambda prompt: next(answers),
        output_func=outputs.append,
        error_func=errors.append,
    )

    assert code == 1
    assert any("lifecycle:RUN_STATE_ERROR" in line for line in errors)


def test_chat_loop_formats_terminal_run_error(tmp_path: Path) -> None:
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=FailingProvider()),  # type: ignore[arg-type]
        store=InMemoryLifecycleStore(),
    )
    failed = threading.Event()
    outputs: list[str] = []
    errors: list[str] = []
    input_index = 0

    def read_input(prompt: str) -> str:
        nonlocal input_index
        del prompt
        input_index += 1
        if input_index == 1:
            return "执行"
        assert failed.wait(1)
        return ""

    def write_error(message: str) -> None:
        errors.append(message)
        failed.set()

    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=read_input,
        output_func=outputs.append,
        error_func=write_error,
    )

    assert code == 1
    assert len(errors) == 1
    assert errors[0].startswith("provider:PROVIDER_ERROR: 调用失败")
