"""基于 ``AgentRunner`` 的多轮交互式 chat CLI。"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..agents import AgentConfig, load_agent_config
from ..config import init_config, is_config_initialized
from ..exceptions import HITLCheckpointInvalidError
from ..harness import (
    AgentRunner,
    AgentRunOptions,
    AgentRunRequest,
    RunLimits,
    RunPhase,
    RunResult,
    RunStopReason,
    RuntimeExecutionOptions,
)
from ..hitl import (
    HumanInteraction,
    PermissionInteractionResponse,
    PermissionPrompt,
    QuestionInteractionResponse,
    QuestionPrompt,
)
from ..providers import create_provider_client
from .render import ChatRenderer
from .trace import ChatTraceStore, TracingRuntimeProvider

TraceMode = Literal["off", "compact", "full"]


@dataclass(slots=True)
class ChatOptions:
    """``iris chat`` 的命令行选项。"""

    config_path: Path
    session_id: str = "cli"
    max_steps: int = 8
    trace_mode: TraceMode = "compact"
    trace_file: Path | None = None
    env_file: Path | None = None
    include_tools: bool = True

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id 不能为空")
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")
        if self.trace_mode not in {"off", "compact", "full"}:
            raise ValueError("trace_mode 必须是 off、compact 或 full")


def run_chat(
    options: ChatOptions,
    *,
    input_func: Callable[[str], str] | None = None,
    renderer: ChatRenderer | None = None,
) -> int:
    """装配 harness 并启动 chat CLI。"""
    chat_renderer = renderer or ChatRenderer()
    try:
        agent_config = _load_configured_agent(options)
        trace_store = ChatTraceStore(options.trace_file)
        provider = TracingRuntimeProvider(
            create_provider_client(
                agent_config.to_model_route(),
                base_url=agent_config.model.base_url,
                timeout=agent_config.model.timeout,
            ),
            trace_store,
        )
        runner = AgentRunner.from_config(
            agent_config,
            config_path=options.config_path,
            provider=provider,
        )
    except Exception as exc:
        chat_renderer.render_error(exc)
        return 1

    chat_renderer.render_header(
        agent_config=agent_config,
        session_id=options.session_id,
        workspace=str(runner.runtime.environment.workspace_root),
        trace_mode=options.trace_mode,
    )
    return run_chat_loop(
        runner=runner,
        agent_config=agent_config,
        options=options,
        trace_store=trace_store,
        renderer=chat_renderer,
        input_func=input_func,
    )


def run_chat_loop(
    *,
    runner: AgentRunner,
    agent_config: AgentConfig,
    options: ChatOptions,
    trace_store: ChatTraceStore,
    renderer: ChatRenderer,
    input_func: Callable[[str], str] | None = None,
) -> int:
    """执行可测试的同步 CLI host loop。"""
    import asyncio

    return asyncio.run(
        _run_chat_loop_async(
            runner=runner,
            agent_config=agent_config,
            options=options,
            trace_store=trace_store,
            renderer=renderer,
            input_func=input_func,
        )
    )


async def _run_chat_loop_async(
    *,
    runner: AgentRunner,
    agent_config: AgentConfig,
    options: ChatOptions,
    trace_store: ChatTraceStore,
    renderer: ChatRenderer,
    input_func: Callable[[str], str] | None = None,
) -> int:
    """在一个 event loop 中依次 start/resume logical runs。"""
    del agent_config
    read_input = input_func or builtins.input
    trace_mode = options.trace_mode
    turn_index = 0

    while True:
        try:
            user_input = read_input("iris> ")
        except KeyboardInterrupt:
            renderer.render_warning("已退出。")
            return 130
        except EOFError:
            return 0

        user_input = user_input.strip()
        if not user_input:
            continue
        command_result = _handle_command(user_input, trace_mode, renderer)
        if command_result == "exit":
            return 0
        if command_result in {"handled", "invalid"}:
            if user_input.startswith("/trace "):
                trace_mode = _parse_trace_mode(user_input, trace_mode)
            continue

        turn_index += 1
        trace_store.start_turn(turn_index)
        renderer.render_user_turn(turn_index, user_input)
        try:
            result = await runner.start(
                AgentRunRequest(
                    input=user_input,
                    session_id=options.session_id,
                ),
                options=AgentRunOptions(
                    limits=RunLimits(max_model_steps=options.max_steps),
                    runtime=RuntimeExecutionOptions(include_tools=options.include_tools),
                ),
            )
            if result.run.phase is RunPhase.WAITING:
                result = await _resume_until_terminal(
                    runner,
                    result,
                    input_func=read_input,
                    renderer=renderer,
                )
        except KeyboardInterrupt:
            renderer.render_warning("人工交互已中断，interaction 保持 pending。")
            return 130
        except EOFError:
            return 0
        except Exception as exc:
            renderer.render_error(exc)
            return 1

        _render_turn_result(
            result,
            turn_index=turn_index,
            trace_mode=trace_mode,
            trace_store=trace_store,
            renderer=renderer,
        )
        if result.run.stop_reason in {
            RunStopReason.FAILED,
            RunStopReason.OUTCOME_UNKNOWN,
        }:
            return 1


def _collect_interaction_response(
    interaction: HumanInteraction,
    *,
    input_func: Callable[[str], str],
    renderer: ChatRenderer,
) -> PermissionInteractionResponse | QuestionInteractionResponse:
    """把终端输入映射为 typed HITL response。"""
    prompt = interaction.request.prompt
    if isinstance(prompt, PermissionPrompt):
        renderer.render_permission_interaction(interaction)
        while True:
            token = input_func("批准该调用？ [y/N] ").strip().lower()
            if token in {"y", "yes"}:
                return PermissionInteractionResponse(decision="approve")
            if token in {"", "n", "no"}:
                return PermissionInteractionResponse(decision="reject")
            renderer.render_warning("请输入 y/yes/n/no；空输入默认拒绝。")
    if isinstance(prompt, QuestionPrompt):
        renderer.render_question_interaction(interaction)
        while True:
            answer = input_func("回答> ").strip()
            if not answer:
                renderer.render_warning("回答不能为空，请重新输入。")
                continue
            if prompt.options and answer.isdecimal():
                option_index = int(answer) - 1
                if 0 <= option_index < len(prompt.options):
                    return QuestionInteractionResponse(answer=prompt.options[option_index])
                renderer.render_warning("请输入有效的选项编号，或输入自由文本。")
                continue
            return QuestionInteractionResponse(answer=answer)
    raise HITLCheckpointInvalidError("interaction prompt 不受支持")


async def _resume_until_terminal(
    runner: AgentRunner,
    result: RunResult,
    *,
    input_func: Callable[[str], str],
    renderer: ChatRenderer,
) -> RunResult:
    """依次处理一个 logical run 的人工 gates。"""
    while result.run.phase is RunPhase.WAITING:
        interaction = result.pending_interaction
        if interaction is None:
            raise HITLCheckpointInvalidError("waiting 结果缺少 pending interaction")
        response = _collect_interaction_response(
            interaction,
            input_func=input_func,
            renderer=renderer,
        )
        result = await runner.resume(
            result.run.run_id,
            interaction_id=interaction.interaction_id,
            response=response,
        )
    return result


def _render_turn_result(
    result: RunResult,
    *,
    turn_index: int,
    trace_mode: TraceMode,
    trace_store: ChatTraceStore,
    renderer: ChatRenderer,
) -> None:
    steps = trace_store.steps_for_turn(turn_index)
    if trace_mode == "compact":
        renderer.render_trace_compact(steps)
    elif trace_mode == "full":
        renderer.render_trace_full(steps)
    for warning in trace_store.warnings:
        renderer.render_warning(warning)
    trace_store.warnings.clear()
    renderer.render_assistant(result)


def _load_configured_agent(options: ChatOptions) -> AgentConfig:
    if not is_config_initialized():
        init_config(env_file=str(options.env_file) if options.env_file is not None else None)
    return load_agent_config(options.config_path)


def _handle_command(user_input: str, trace_mode: TraceMode, renderer: ChatRenderer) -> str:
    if not user_input.startswith("/"):
        return "none"
    if user_input in {"/exit", "/quit"}:
        return "exit"
    if user_input == "/help":
        renderer.render_help(trace_mode)
        return "handled"
    if user_input.startswith("/trace "):
        next_mode = _parse_trace_mode(user_input, trace_mode)
        if next_mode == trace_mode and user_input.split(maxsplit=1)[1] not in {
            "off",
            "compact",
            "full",
        }:
            renderer.render_warning("用法: /trace off|compact|full")
            return "invalid"
        renderer.render_warning(f"trace 已切换为 {next_mode}")
        return "handled"
    renderer.render_warning("未知命令。输入 /help 查看可用命令。")
    return "invalid"


def _parse_trace_mode(user_input: str, current: TraceMode) -> TraceMode:
    parts = user_input.split(maxsplit=1)
    if len(parts) != 2:
        return current
    value = parts[1]
    if value in {"off", "compact", "full"}:
        return value  # type: ignore[return-value]
    return current


__all__ = ["ChatOptions", "run_chat", "run_chat_loop"]
