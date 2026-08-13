from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from iris.cli.chat import ChatOptions, run_chat_loop
from iris.exceptions import IrisRunStateError
from iris.harness import (
    AgentRunOptions,
    AgentRunRequest,
    RunLimits,
    RuntimeExecutionOptions,
)
from iris.hitl import (
    HumanInteraction,
    HumanInteractionRequest,
    HumanInteractionResponse,
    InteractionStatus,
    PermissionInteractionResponse,
    PermissionPrompt,
    QuestionInteractionResponse,
    QuestionPrompt,
    ToolCallSnapshot,
)
from iris.lifecycle import (
    RunErrorInfo,
    RunPhase,
    RunResult,
    RunSnapshot,
    RunStopReason,
    RunUsage,
)
from iris.message import Msg


def _snapshot(*, phase: RunPhase, pending_id: str | None = None) -> RunSnapshot:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return RunSnapshot(
        run_id="run-1",
        session_id="cli",
        agent_id="cli-agent",
        phase=phase,
        stop_reason=RunStopReason.COMPLETED if phase is RunPhase.TERMINAL else None,
        revision=1,
        current_activation_id="act-1" if phase is RunPhase.ACTIVE else None,
        pending_interaction_id=pending_id,
        limits=RunLimits(),
        usage=RunUsage(),
        environment_fingerprint="f" * 64,
        checkpoint_sequence=1,
        last_event_sequence=1,
        created_at=now,
        started_at=now,
        updated_at=now,
        finished_at=now if phase is RunPhase.TERMINAL else None,
    )


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


def _waiting(interaction: HumanInteraction) -> RunResult:
    return RunResult(
        run=_snapshot(
            phase=RunPhase.WAITING,
            pending_id=interaction.interaction_id,
        ),
        pending_interaction=interaction,
    )


def _terminal(text: str = "完成") -> RunResult:
    return RunResult(
        run=_snapshot(phase=RunPhase.TERMINAL),
        assistant_message=Msg.assistant(text),
    )


class SequenceRunner:
    def __init__(self, initial: RunResult, resumed: RunResult | None = None) -> None:
        self.initial = initial
        self.resumed = resumed
        self.start_calls: list[tuple[AgentRunRequest, AgentRunOptions]] = []
        self.resume_calls: list[
            tuple[str, str, HumanInteractionResponse]
        ] = []
        self.loop_ids: list[int] = []

    async def start(
        self,
        request: AgentRunRequest,
        *,
        options: AgentRunOptions,
    ) -> RunResult:
        self.start_calls.append((request, options))
        self.loop_ids.append(id(asyncio.get_running_loop()))
        return self.initial

    async def resume(
        self,
        run_id: str,
        *,
        interaction_id: str,
        response: HumanInteractionResponse,
    ) -> RunResult:
        self.resume_calls.append((run_id, interaction_id, response))
        self.loop_ids.append(id(asyncio.get_running_loop()))
        assert self.resumed is not None
        return self.resumed


class ErrorRunner:
    async def start(self, request: object, *, options: object) -> RunResult:
        del request, options
        raise IrisRunStateError("状态不允许")


def test_chat_loop_starts_and_resumes_in_one_event_loop(tmp_path: Path) -> None:
    interaction = _interaction(PermissionPrompt(reason="写入"))
    runner = SequenceRunner(_waiting(interaction), _terminal("已完成"))
    answers = iter(["执行", "y", "/exit"])
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
    assert runner.start_calls == [
        (
            AgentRunRequest(input="执行", session_id="cli"),
            AgentRunOptions(
                limits=RunLimits(max_model_steps=8),
                runtime=RuntimeExecutionOptions(include_tools=True),
            ),
        )
    ]
    assert runner.resume_calls == [
        (
            "run-1",
            "interaction-1",
            PermissionInteractionResponse(decision="approve"),
        )
    ]
    assert len(set(runner.loop_ids)) == 1
    assert any("已完成" in line for line in outputs)
    assert errors == []


def test_chat_loop_maps_second_question_option(tmp_path: Path) -> None:
    interaction = _interaction(
        QuestionPrompt(question="选择格式", options=["Markdown", "纯文本"])
    )
    runner = SequenceRunner(_waiting(interaction), _terminal())
    answers = iter(["执行", "2", "/exit"])
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
    assert runner.resume_calls == [
        (
            "run-1",
            "interaction-1",
            QuestionInteractionResponse(answer="纯文本"),
        )
    ]
    assert any("1. Markdown" in line for line in outputs)
    assert any("2. 纯文本" in line for line in outputs)
    assert errors == []


def test_chat_loop_accepts_free_text_question_answer(tmp_path: Path) -> None:
    interaction = _interaction(
        QuestionPrompt(question="补充说明", options=["跳过", "继续"])
    )
    runner = SequenceRunner(_waiting(interaction), _terminal())
    answers = iter(["执行", "自定义回答", "/exit"])
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
    assert runner.resume_calls == [
        (
            "run-1",
            "interaction-1",
            QuestionInteractionResponse(answer="自定义回答"),
        )
    ]
    assert errors == []


def test_chat_loop_displays_permission_details(tmp_path: Path) -> None:
    interaction = _interaction(PermissionPrompt(reason="写入笔记"))
    runner = SequenceRunner(_waiting(interaction), _terminal())
    answers = iter(["执行", "n", "/exit"])
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
    assert "write_file" in output
    assert "note.txt" in output
    assert "写入笔记" in output
    assert "本次批准只适用于该调用" in output
    assert runner.resume_calls == [
        (
            "run-1",
            "interaction-1",
            PermissionInteractionResponse(decision="reject"),
        )
    ]
    assert errors == []


def test_chat_loop_help_lists_exit_and_quit_without_trace(tmp_path: Path) -> None:
    runner = SequenceRunner(_terminal())
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
    runner = SequenceRunner(_terminal())
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
    runner = SequenceRunner(_terminal())
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
    runner = SequenceRunner(_terminal())
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
    failed_snapshot = _snapshot(phase=RunPhase.TERMINAL).model_copy(
        update={"stop_reason": RunStopReason.FAILED}
    )
    runner = SequenceRunner(
        RunResult(
            run=failed_snapshot,
            error=RunErrorInfo(
                source="provider",
                code="PROVIDER_ERROR",
                message="调用失败",
            ),
        )
    )
    answers = iter(["执行"])
    outputs: list[str] = []
    errors: list[str] = []

    code = run_chat_loop(
        runner=runner,  # type: ignore[arg-type]
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=lambda prompt: next(answers),
        output_func=outputs.append,
        error_func=errors.append,
    )

    assert code == 1
    assert errors == ["provider:PROVIDER_ERROR: 调用失败"]
