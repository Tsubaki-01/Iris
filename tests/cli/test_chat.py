from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rich.console import Console

import iris.cli.chat as chat_module
from iris.agents import AgentConfig
from iris.cli.chat import ChatOptions, run_chat_loop
from iris.cli.render import ChatRenderer
from iris.cli.trace import ChatTraceStore
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
from iris.lifecycle import (
    RunLimits,
    RunPhase,
    RunResult,
    RunSnapshot,
    RunStopReason,
    RunUsage,
)
from iris.message import Msg


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="cli-agent",
        model={"provider": "openai", "name": "fake-model"},
        system="测试",
    )


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
        tool_name="tool",
        arguments={},
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
        run=_snapshot(phase=RunPhase.WAITING, pending_id=interaction.interaction_id),
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
        self.start_inputs: list[str] = []
        self.resume_responses: list[object] = []
        self.loop_ids: list[int] = []

    async def start(self, request: object, *, options: object) -> RunResult:
        del options
        self.start_inputs.append(request.input)
        self.loop_ids.append(id(asyncio.get_running_loop()))
        return self.initial

    async def resume(
        self,
        run_id: str,
        *,
        interaction_id: str,
        response: object,
    ) -> RunResult:
        del run_id, interaction_id
        self.resume_responses.append(response)
        self.loop_ids.append(id(asyncio.get_running_loop()))
        assert self.resumed is not None
        return self.resumed


def _renderer() -> tuple[ChatRenderer, Console]:
    console = Console(record=True, width=120, color_system=None)
    return ChatRenderer(console), console


@pytest.mark.parametrize("token", ["y", "yes"])
def test_collect_permission_response_approves_supported_tokens(token: str) -> None:
    renderer, _ = _renderer()

    response = chat_module._collect_interaction_response(
        _interaction(PermissionPrompt(reason="写入")),
        input_func=lambda _: token,
        renderer=renderer,
    )

    assert response == PermissionInteractionResponse(decision="approve")


def test_collect_question_response_maps_option_number() -> None:
    renderer, _ = _renderer()

    response = chat_module._collect_interaction_response(
        _interaction(QuestionPrompt(question="环境？", options=["测试", "生产"])),
        input_func=lambda _: "2",
        renderer=renderer,
    )

    assert response == QuestionInteractionResponse(answer="生产")


def test_chat_loop_uses_agent_runner_start_and_resume_in_one_event_loop(tmp_path: Path) -> None:
    interaction = _interaction(PermissionPrompt(reason="写入"))
    runner = SequenceRunner(_waiting(interaction), _terminal("已完成"))
    renderer, console = _renderer()
    answers = iter(["执行", "y", "/exit"])

    code = run_chat_loop(
        runner=runner,  # type: ignore[arg-type]
        agent_config=_agent_config(),
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        trace_store=ChatTraceStore(),
        renderer=renderer,
        input_func=lambda _: next(answers),
    )

    assert code == 0
    assert runner.start_inputs == ["执行"]
    assert runner.resume_responses == [PermissionInteractionResponse(decision="approve")]
    assert len(set(runner.loop_ids)) == 1
    assert "已完成" in console.export_text()


def test_chat_options_validate_session_steps_and_trace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="session_id"):
        ChatOptions(config_path=tmp_path / "agent.yaml", session_id=" ")
    with pytest.raises(ValueError, match="max_steps"):
        ChatOptions(config_path=tmp_path / "agent.yaml", max_steps=0)
    with pytest.raises(ValueError, match="trace_mode"):
        ChatOptions(config_path=tmp_path / "agent.yaml", trace_mode="bad")  # type: ignore[arg-type]


def test_renderer_shows_structured_run_error() -> None:
    result = _terminal().model_copy(
        update={
            "run": _snapshot(phase=RunPhase.TERMINAL).model_copy(
                update={"stop_reason": RunStopReason.FAILED}
            )
        }
    )
    renderer, console = _renderer()

    renderer.render_assistant(result)

    assert "完成" in console.export_text()
