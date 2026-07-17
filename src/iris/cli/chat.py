"""多轮交互式 chat CLI。"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..agents import AgentConfig, load_agent_config
from ..config import init_config, is_config_initialized
from ..exceptions import HITLCheckpointInvalidError
from ..hitl import (
    HumanInteraction,
    InteractionKind,
    InteractionStatus,
    PermissionInteractionRequest,
    PermissionInteractionResponse,
    QuestionInteractionRequest,
    QuestionInteractionResponse,
)
from ..providers import create_provider_client
from ..runtime import AgentRuntime, RuntimeFactory
from ..runtime.models import BoundedLoopOptions, RuntimeOptions, RuntimeStatus, RuntimeTurnResult
from .render import ChatRenderer
from .trace import ChatTraceStore, TracingRuntimeProvider

TraceMode = Literal["off", "compact", "full"]


@dataclass(slots=True)
class ChatOptions:
    """`iris chat` 的命令行选项。"""

    config_path: Path
    session_id: str = "cli"
    max_steps: int = 8
    trace_mode: TraceMode = "compact"
    trace_file: Path | None = None
    env_file: Path | None = None
    include_tools: bool = True

    def __post_init__(self) -> None:
        """校验命令选项。"""
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
    """启动多轮 chat CLI。

    Args:
        options: chat 选项。
        input_func: 输入函数，测试可注入。
        renderer: Rich 渲染器，测试可注入。

    Returns:
        int: 进程退出码。
    """
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
        runtime = RuntimeFactory.from_config(
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
        workspace=str(runtime.workspace_root),
        trace_mode=options.trace_mode,
    )
    return run_chat_loop(
        runtime=runtime,
        agent_config=agent_config,
        options=options,
        trace_store=trace_store,
        renderer=chat_renderer,
        input_func=input_func,
    )


def run_chat_loop(
    *,
    runtime: AgentRuntime,
    agent_config: AgentConfig,
    options: ChatOptions,
    trace_store: ChatTraceStore,
    renderer: ChatRenderer,
    input_func: Callable[[str], str] | None = None,
) -> int:
    """执行可测试的 chat 输入循环。"""
    import asyncio

    return asyncio.run(
        _run_chat_loop_async(
            runtime=runtime,
            agent_config=agent_config,
            options=options,
            trace_store=trace_store,
            renderer=renderer,
            input_func=input_func,
        )
    )


async def _run_chat_loop_async(
    *,
    runtime: AgentRuntime,
    agent_config: AgentConfig,
    options: ChatOptions,
    trace_store: ChatTraceStore,
    renderer: ChatRenderer,
    input_func: Callable[[str], str] | None = None,
) -> int:
    """在单个 event loop 中执行 chat 输入循环。"""
    del agent_config
    read_input = input_func or builtins.input
    trace_mode = options.trace_mode
    turn_index = 0
    trace_store.start_turn(turn_index)
    try:
        recovered = await _recover_session_if_needed(
            runtime,
            session_id=options.session_id,
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
    if recovered is not None:
        _render_turn_result(
            recovered,
            turn_index=turn_index,
            trace_mode=trace_mode,
            trace_store=trace_store,
            renderer=renderer,
        )
        if recovered.status is RuntimeStatus.ERROR:
            return 1

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
        result = await _run_loop_async(runtime, user_input, options)
        resumed = result.status is RuntimeStatus.WAITING_HUMAN
        if resumed:
            try:
                result = await _resume_until_terminal(
                    runtime,
                    result,
                    input_func=read_input,
                    renderer=renderer,
                )
            except KeyboardInterrupt:
                renderer.render_warning("人工交互已中断，interaction 保持 pending。")
                return 130
            except EOFError:
                return 0
        _render_turn_result(
            result,
            turn_index=turn_index,
            trace_mode=trace_mode,
            trace_store=trace_store,
            renderer=renderer,
        )
        if resumed and result.status is RuntimeStatus.ERROR:
            return 1


def _collect_interaction_response(
    interaction: HumanInteraction,
    *,
    input_func: Callable[[str], str],
    renderer: ChatRenderer,
) -> PermissionInteractionResponse | QuestionInteractionResponse:
    """渲染人工请求并把终端输入映射为现有 typed response。"""
    request = interaction.request
    if interaction.kind is InteractionKind.PERMISSION and isinstance(
        request, PermissionInteractionRequest
    ):
        renderer.render_permission_interaction(interaction)
        while True:
            token = input_func("批准该调用？ [y/N] ").strip().lower()
            if token in {"y", "yes"}:
                return PermissionInteractionResponse(decision="approve")
            if token in {"", "n", "no"}:
                return PermissionInteractionResponse(decision="reject")
            renderer.render_warning("请输入 y/yes/n/no；空输入默认拒绝。")

    if interaction.kind is InteractionKind.QUESTION and isinstance(
        request, QuestionInteractionRequest
    ):
        renderer.render_question_interaction(interaction)
        while True:
            answer = input_func("回答> ").strip()
            if not answer:
                renderer.render_warning("回答不能为空，请重新输入。")
                continue
            if request.options and answer.isdecimal():
                option_index = int(answer) - 1
                if 0 <= option_index < len(request.options):
                    return QuestionInteractionResponse(answer=request.options[option_index])
                renderer.render_warning("请输入有效的选项编号，或输入自由文本。")
                continue
            return QuestionInteractionResponse(answer=answer)

    raise HITLCheckpointInvalidError("interaction kind/request 不匹配")


async def _resume_until_terminal(
    runtime: AgentRuntime,
    result: RuntimeTurnResult,
    *,
    input_func: Callable[[str], str],
    renderer: ChatRenderer,
) -> RuntimeTurnResult:
    """依次处理当前 run 的人工 gate，直到 runtime 返回终态。"""
    while result.status is RuntimeStatus.WAITING_HUMAN:
        interaction = result.pending_interaction
        if interaction is None:
            raise HITLCheckpointInvalidError("waiting_human 结果缺少 pending interaction")
        response = _collect_interaction_response(
            interaction,
            input_func=input_func,
            renderer=renderer,
        )
        result = await runtime.resume(interaction.interaction_id, response)
    return result


async def _recover_session_if_needed(
    runtime: AgentRuntime,
    *,
    session_id: str,
    input_func: Callable[[str], str],
    renderer: ChatRenderer,
) -> RuntimeTurnResult | None:
    """在读取普通输入前恢复当前 session 的 active HITL run。"""
    interaction = runtime.load_resumable_interaction(session_id)
    if interaction is None:
        return None
    renderer.render_recovery_notice(interaction)
    response = None
    if interaction.status is InteractionStatus.PENDING:
        response = _collect_interaction_response(
            interaction,
            input_func=input_func,
            renderer=renderer,
        )
    result = await runtime.resume(interaction.interaction_id, response)
    return await _resume_until_terminal(
        runtime,
        result,
        input_func=input_func,
        renderer=renderer,
    )


def _render_turn_result(
    result: RuntimeTurnResult,
    *,
    turn_index: int,
    trace_mode: TraceMode,
    trace_store: ChatTraceStore,
    renderer: ChatRenderer,
) -> None:
    """按既有顺序渲染一次 terminal runtime 结果。"""
    steps = trace_store.steps_for_turn(turn_index)
    if trace_mode == "compact":
        renderer.render_trace_compact(steps)
    elif trace_mode == "full":
        renderer.render_trace_full(steps)
    for warning in trace_store.warnings:
        renderer.render_warning(warning)
    trace_store.warnings.clear()
    renderer.render_tool_results(result)
    renderer.render_assistant(result)


def _load_configured_agent(options: ChatOptions) -> AgentConfig:
    """初始化全局配置并读取 agent YAML。"""
    if not is_config_initialized():
        init_config(env_file=str(options.env_file) if options.env_file is not None else None)
    return load_agent_config(options.config_path)


async def _run_loop_async(
    runtime: AgentRuntime,
    user_input: str,
    options: ChatOptions,
) -> RuntimeTurnResult:
    """调用 runtime.run_loop 并复用 chat 进程的 event loop。"""
    return await runtime.run_loop(
        user_input,
        options=RuntimeOptions(
            session_id=options.session_id,
            include_tools=options.include_tools,
            loop=BoundedLoopOptions(max_steps=options.max_steps),
        ),
    )


def _handle_command(
    user_input: str,
    trace_mode: TraceMode,
    renderer: ChatRenderer,
) -> str:
    """处理 slash command。"""
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
    """从 slash command 中解析 trace mode。"""
    parts = user_input.split(maxsplit=1)
    if len(parts) != 2:
        return current
    value = parts[1]
    if value in {"off", "compact", "full"}:
        return value  # type: ignore[return-value]
    return current


__all__ = ["ChatOptions", "run_chat", "run_chat_loop"]
