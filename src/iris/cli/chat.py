"""多轮交互式 chat CLI。"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..agents import AgentConfig, load_agent_config
from ..config import init_config, is_config_initialized
from ..providers import create_provider_client
from ..runtime import AgentRuntime, RuntimeFactory
from ..runtime.models import BoundedLoopOptions, RuntimeOptions, RuntimeTurnResult
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
        result = _run_loop_sync(runtime, user_input, options)
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
        init_config(
            env_file=str(options.env_file) if options.env_file is not None else None
        )
    return load_agent_config(options.config_path)


def _run_loop_sync(
    runtime: AgentRuntime,
    user_input: str,
    options: ChatOptions,
) -> RuntimeTurnResult:
    """同步调用 runtime.run_loop，便于隔离 asyncio.run。"""
    import asyncio

    return asyncio.run(
        runtime.run_loop(
            user_input,
            options=RuntimeOptions(
                session_id=options.session_id,
                include_tools=options.include_tools,
                loop=BoundedLoopOptions(max_steps=options.max_steps),
            ),
        )
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
