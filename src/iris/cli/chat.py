"""基于 ``SessionManager`` 的标准库交互式 chat CLI。

Example:
    options = ChatOptions(config_path=Path("agent.yaml"))
    exit_code = run_chat(options)
"""

# region imports
from __future__ import annotations

import asyncio
import builtins
import json
import sys
import threading
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..config import init_config, is_config_initialized
from ..exceptions import HITLCheckpointInvalidError, IrisError
from ..harness import (
    AgentRunner,
    AgentRunOptions,
    RunEventKind,
    RunLimits,
    RunResult,
    RunStopReason,
    RuntimeExecutionOptions,
    SessionManager,
    SubmissionEvent,
)
from ..harness.streaming import LiveFact
from ..hitl import (
    HumanInteraction,
    PermissionInteractionResponse,
    PermissionPrompt,
    QuestionInteractionResponse,
    QuestionPrompt,
)
from ..message import (
    ModelBlockDelta,
    ModelResponseCancelled,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelResponseStarted,
)
from ..runtime import RuntimeStreamEvent

# endregion


@dataclass(slots=True)
class ChatOptions:
    """``iris chat`` 的命令行选项。

    Attributes:
        config_path (Path): Agent YAML 配置路径。
        session_id (str): lifecycle 会话标识。
        max_steps (int): 每轮最多允许的模型步数。
        env_file (Path | None): 可选 dotenv 文件路径。
        include_tools (bool): 是否向 provider 暴露工具。

    Example:
        options = ChatOptions(config_path=Path("agent.yaml"), max_steps=4)
        assert options.max_steps == 4
    """

    config_path: Path
    session_id: str = "cli"
    max_steps: int = 8
    env_file: Path | None = None
    include_tools: bool = True

    def __post_init__(self) -> None:
        """校验 chat 选项。

        Raises:
            ValueError: 会话标识为空或模型步数不是正数。
        """
        if not self.session_id.strip():
            raise ValueError("session_id 不能为空")
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")


def run_chat(
    options: ChatOptions,
    *,
    input_func: Callable[[str], str] | None = None,
    output_func: Callable[[str], None] | None = None,
    error_func: Callable[[str], None] | None = None,
) -> int:
    """装配带 live streaming 的 complete-run harness 并启动 chat。

    Args:
        options (ChatOptions): chat 命令选项。
        input_func (Callable[[str], str] | None): 可选输入回调。
        output_func (Callable[[str], None] | None): 可选标准输出回调。
        error_func (Callable[[str], None] | None): 可选标准错误回调。

    Returns:
        int: 进程退出码。
    """
    write_error = error_func or (lambda message: print(message, file=sys.stderr))
    try:
        if not is_config_initialized():
            init_config(env_file=str(options.env_file) if options.env_file is not None else None)
        live_output = _ChatLiveOutput(
            output_func or (lambda fragment: print(fragment, end="", flush=True))
        )
        runner = AgentRunner.from_config_path(
            options.config_path,
            live_publisher=live_output,
        )
    except IrisError as exc:
        write_error(_format_iris_error(exc))
        return 1

    return run_chat_loop(
        runner=runner,
        options=options,
        live_output=live_output,
        input_func=input_func,
        output_func=output_func,
        error_func=write_error,
    )


def run_chat_loop(
    *,
    runner: AgentRunner,
    options: ChatOptions,
    live_output: _ChatLiveOutput | None = None,
    input_func: Callable[[str], str] | None = None,
    output_func: Callable[[str], None] | None = None,
    error_func: Callable[[str], None] | None = None,
) -> int:
    """在主线程读取终端输入，并在后台 event loop 推进 session。

    ``input()`` 保留在主线程，使 Ctrl-C 继续表现为同步 ``KeyboardInterrupt``；runner 与
    ``SessionManager`` 在一个后台 event loop 中运行，因此 provider 执行期间仍可接收输入。

    Args:
        runner (AgentRunner): complete-run SDK facade。
        options (ChatOptions): chat 命令选项。
        live_output (_ChatLiveOutput | None): 与 runner 共享的可选同步文本输出。
        input_func (Callable[[str], str] | None): 可选输入回调。
        output_func (Callable[[str], None] | None): 可选标准输出回调。
        error_func (Callable[[str], None] | None): 可选标准错误回调。

    Returns:
        int: 进程退出码。
    """
    read_input = input_func or builtins.input
    write_output = output_func or builtins.print
    write_error = error_func or (lambda message: print(message, file=sys.stderr))
    host = _ChatSessionHost(
        runner=runner,
        options=options,
        live_output=live_output,
        output_func=write_output,
        error_func=write_error,
    )
    host.start()
    close_reason = "chat 结束"
    exit_code = 0
    try:
        while True:
            try:
                user_input = read_input("iris> ").strip()
            except KeyboardInterrupt:
                close_reason = "用户中断"
                exit_code = 130
                break
            except EOFError:
                close_reason = "输入已关闭"
                break

            if host.exit_code is not None:
                exit_code = host.exit_code
                break
            if user_input in {"/exit", "/quit"}:
                close_reason = "用户退出 chat"
                break
            if user_input == "/help":
                write_output("可用命令：")
                write_output("/follow-up <消息>  排入下一轮")
                write_output("/help  显示帮助")
                write_output("/exit  退出 chat")
                write_output("/quit  退出 chat")
                continue
            if user_input == "/follow-up":
                write_output("用法：/follow-up <消息>")
                continue
            if user_input.startswith("/follow-up "):
                follow_up = user_input.removeprefix("/follow-up ").strip()
                if not follow_up:
                    write_output("用法：/follow-up <消息>")
                    continue
                host.submit(follow_up, mode="follow_up")
                continue
            if user_input.startswith("/"):
                write_output("未知命令。输入 /help 查看可用命令。")
                continue

            host.submit(user_input)
    except IrisError as exc:
        write_error(_format_iris_error(exc))
        exit_code = 1
    finally:
        try:
            host.close(reason=close_reason)
        except IrisError as exc:
            write_error(_format_iris_error(exc))
            exit_code = 1
    return exit_code


class _ChatLiveOutput:
    """在 runner 所属 event loop 中同步显示模型文本，保留逐 run 收尾标记。"""

    def __init__(self, write: Callable[[str], None]) -> None:
        """保存文本回调；显示状态只在同一个 event loop 中读写。"""
        self._write = write
        self._open_runs: set[str] = set()
        self._completed_runs: set[str] = set()

    def publish(self, fact: LiveFact) -> None:
        """消费模型文本事件；durable 与 submission 事实由 manager 处理。"""
        if not isinstance(fact, RuntimeStreamEvent):
            return
        event = fact.model_event
        if isinstance(event, ModelResponseStarted):
            self._completed_runs.discard(fact.run_id)
        elif isinstance(event, ModelBlockDelta) and event.channel == "text":
            self._write(event.delta)
            self._open_runs.add(fact.run_id)
        elif isinstance(
            event, (ModelResponseCompleted, ModelResponseFailed, ModelResponseCancelled)
        ):
            displayed = fact.run_id in self._open_runs
            self._finish_text(fact.run_id)
            if displayed and isinstance(event, ModelResponseCompleted):
                self._completed_runs.add(fact.run_id)
            else:
                self._completed_runs.discard(fact.run_id)

    def finish_run(self, run_id: str) -> bool:
        """收尾并清理该 run，返回完整模型文本是否已经展示。"""
        self._finish_text(run_id)
        displayed = run_id in self._completed_runs
        self._completed_runs.discard(run_id)
        return displayed

    def close(self) -> None:
        """关闭 host 时收尾尚无模型终态的文本行，并释放完成标记。"""
        for run_id in tuple(self._open_runs):
            self._finish_text(run_id)
        self._completed_runs.clear()

    def _finish_text(self, run_id: str) -> None:
        """只为已经输出文本的模型调用补一次换行。"""
        if run_id in self._open_runs:
            self._write("\n")
            self._open_runs.remove(run_id)


class _ChatSessionHost:
    """把同步终端输入桥接到单 session 的异步 facade。

    一个实例只绑定一个 runner、一个 session id 和一个后台 event loop。主线程通过同步方法
    提交输入；所有 manager 状态、事件和 HITL resume 都留在后台 loop 内。

    Attributes:
        _runner (AgentRunner): durable complete-run owner。
        _options (ChatOptions): CLI 会话与 run 选项。
        _live_output (_ChatLiveOutput | None): Runner 与 CLI 共享的同步文本输出。
        _output_func (Callable[[str], None]): 标准输出回调。
        _error_func (Callable[[str], None]): 标准错误回调。
        _thread (threading.Thread): 承载 asyncio event loop 的后台线程。
        _ready (threading.Event): 后台 host 已可接收调用的同步点。
        _loop (asyncio.AbstractEventLoop | None): manager 所属 event loop。
        _stop (asyncio.Event | None): 请求关闭 event stream 的信号。
        _manager (SessionManager | None): 当前 CLI 使用的单 session facade。
        _pending_interaction (HumanInteraction | None): 等待下一行输入的 typed HITL。
        _close_reason (str | None): 退出时交给 runner 的取消原因。
        _exit_code (int | None): terminal failure 请求的 CLI 退出码。
        _thread_error (BaseException | None): 后台 host 的未处理错误。

    Example:
        host = _ChatSessionHost(
            runner=runner,
            options=options,
            live_output=None,
            output_func=print,
            error_func=print,
        )
        host.start()
        host.submit("你好")
        host.close()
    """

    # ==========================================
    #               Initialization
    # ==========================================
    # region
    def __init__(
        self,
        runner: AgentRunner,
        options: ChatOptions,
        live_output: _ChatLiveOutput | None,
        output_func: Callable[[str], None],
        error_func: Callable[[str], None],
    ) -> None:
        """保存 host 依赖；异步资源由后台线程创建。"""
        self._runner = runner
        self._options = options
        self._live_output = live_output
        self._output_func = output_func
        self._error_func = error_func
        self._thread = threading.Thread(target=self._run, name="iris-chat-host")
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._manager: SessionManager | None = None
        self._pending_interaction: HumanInteraction | None = None
        self._close_reason: str | None = None
        self._exit_code: int | None = None
        self._thread_error: BaseException | None = None

    # endregion

    # ==========================================
    #                Public API
    # ==========================================
    # region
    @property
    def exit_code(self) -> int | None:
        """返回后台 terminal failure 请求的退出码。"""
        return self._exit_code

    def start(self) -> None:
        """启动后台 event loop，并等待 manager 可用。"""
        self._thread.start()
        self._ready.wait()
        if self._thread_error is not None:
            raise self._thread_error

    def submit(self, input: str, *, mode: Literal["follow_up"] | None = None) -> None:
        """同步提交一行普通输入或 follow-up 命令。"""
        self._call(self._submit(input, mode=mode))

    def close(self, *, reason: str | None = None) -> None:
        """停止 admission 并等待当前 run 结算后结束后台线程。"""
        if not self._thread.is_alive():
            return
        loop = self._require_loop()
        stop = self._require_stop()
        self._close_reason = reason
        loop.call_soon_threadsafe(stop.set)
        self._thread.join()
        if self._thread_error is not None:
            raise self._thread_error

    # endregion

    # ==========================================
    #              Async Session Flow
    # ==========================================
    # region
    def _run(self) -> None:
        """在线程内建立并完整关闭 asyncio event loop。"""
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self._thread_error = exc
            self._ready.set()

    async def _serve(self) -> None:
        """在后台 loop 消费事件，原调用结束后依次关闭 runner 与输出。"""
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        self._manager = SessionManager(
            self._runner,
            self._options.session_id,
        )
        consumer = asyncio.create_task(self._consume_events())
        self._ready.set()
        try:
            await self._stop.wait()
        finally:
            failure = sys.exception()
            try:
                # 三步依次完成；manager 等原调用结束后才允许关闭 root MCP 资源。
                for operation in (
                    self._manager.close(cancel_run=True, reason=self._close_reason),
                    self._runner.aclose(),
                    consumer,
                ):
                    try:
                        await operation
                    except BaseException as error:
                        if failure is None:
                            failure = error
                        else:
                            self._error_func(
                                _format_iris_error(error)
                                if isinstance(error, IrisError)
                                else str(error)
                            )
            finally:
                if self._live_output is not None:
                    self._live_output.close()
            if failure is not None:
                raise failure

    async def _submit(
        self,
        input: str,
        *,
        mode: Literal["follow_up"] | None,
    ) -> None:
        """按当前 host 状态路由一行输入。"""
        manager = self._require_manager()
        if mode == "follow_up":
            await manager.submit(
                input,
                mode="follow_up",
                options=self._run_options(),
            )
            return

        if self._pending_interaction is not None:
            response = _parse_interaction_response(
                self._pending_interaction,
                input,
                output_func=self._output_func,
            )
            if response is None:
                return
            interaction = self._pending_interaction
            self._pending_interaction = None
            await manager.admit_resume(
                interaction_id=interaction.interaction_id,
                response=response,
            )
            return

        if not input.strip():
            return
        await manager.submit(
            input,
            mode="auto",
            options=self._run_options(),
        )

    async def _consume_events(self) -> None:
        """消费 mixed stream，显示结果及仍待响应的交互。"""
        async for event in self._require_manager().events():
            if isinstance(event, SubmissionEvent):
                continue
            if event.kind is RunEventKind.INTERACTION_SUSPENDED:
                result = self._runner.get_result(event.run_id)
                # 输入控制读取当前事实；历史 suspended 提示可能已被取消或恢复取代。
                if (
                    result is None
                    or result.pending_interaction is None
                    or result.pending_interaction.interaction_id != event.correlation_id
                ):
                    continue
                self._pending_interaction = result.pending_interaction
                _write_interaction_prompt(
                    result.pending_interaction,
                    output_func=self._output_func,
                )
                continue
            if event.kind is RunEventKind.RUN_TERMINAL:
                result = self._runner.get_result(event.run_id)
                if result is None:
                    raise HITLCheckpointInvalidError("terminal 事件缺少 durable result")
                _write_result(
                    result,
                    output_func=self._output_func,
                    error_func=self._error_func,
                    include_assistant=(
                        self._live_output is None or not self._live_output.finish_run(event.run_id)
                    ),
                )
                if result.run.stop_reason in {
                    RunStopReason.FAILED,
                    RunStopReason.OUTCOME_UNKNOWN,
                }:
                    self._exit_code = 1
                if (
                    self._pending_interaction is not None
                    and self._pending_interaction.run_id == event.run_id
                ):
                    self._pending_interaction = None

    # endregion

    # ==========================================
    #                Helpers
    # ==========================================
    # region
    def _call[T](self, coroutine: Coroutine[Any, Any, T]) -> T:
        """在 manager event loop 中执行 coroutine，并同步返回结果。"""
        future = asyncio.run_coroutine_threadsafe(coroutine, self._require_loop())
        return future.result()

    def _run_options(self) -> AgentRunOptions:
        """把 CLI options 投影为每个新 run 使用的固定 options。"""
        return AgentRunOptions(
            limits=RunLimits(max_model_steps=self._options.max_steps),
            runtime=RuntimeExecutionOptions(include_tools=self._options.include_tools),
        )

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        """返回已启动的后台 event loop。"""
        assert self._loop is not None
        return self._loop

    def _require_stop(self) -> asyncio.Event:
        """返回已启动的 stop signal。"""
        assert self._stop is not None
        return self._stop

    def _require_manager(self) -> SessionManager:
        """返回已启动的 session manager。"""
        assert self._manager is not None
        return self._manager

    # endregion


def _write_interaction_prompt(
    interaction: HumanInteraction,
    *,
    output_func: Callable[[str], None],
) -> None:
    """把 typed HITL prompt 展示到终端，但不读取输入。

    Args:
        interaction (HumanInteraction): 当前 pending interaction。
        output_func (Callable[[str], None]): 标准输出回调。

    Raises:
        HITLCheckpointInvalidError: prompt 类型不受支持。
    """
    prompt = interaction.request.prompt
    if isinstance(prompt, PermissionPrompt):
        tool_call = interaction.request.tool_call
        output_func(f"工具: {tool_call.tool_name}")
        output_func(
            "参数: "
            + json.dumps(
                tool_call.arguments,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        output_func(f"原因: {prompt.reason}")
        output_func("本次批准只适用于该调用。")
        output_func("批准该调用？ [y/N]")
        return
    if isinstance(prompt, QuestionPrompt):
        output_func(prompt.question)
        for index, option in enumerate(prompt.options, start=1):
            output_func(f"{index}. {option}")
        output_func("请输入回答。")
        return
    raise HITLCheckpointInvalidError("interaction prompt 不受支持")


def _parse_interaction_response(
    interaction: HumanInteraction,
    input: str,
    *,
    output_func: Callable[[str], None],
) -> PermissionInteractionResponse | QuestionInteractionResponse | None:
    """把单行终端输入映射为 typed HITL response。

    Args:
        interaction (HumanInteraction): 当前 pending interaction。
        input (str): 用户本次输入的原始文本。
        output_func (Callable[[str], None]): 校验失败提示回调。

    Returns:
        PermissionInteractionResponse | QuestionInteractionResponse | None:
            输入有效时返回 typed response；需要重新输入时返回 None。

    Raises:
        HITLCheckpointInvalidError: prompt 类型不受支持。
    """
    prompt = interaction.request.prompt
    if isinstance(prompt, PermissionPrompt):
        token = input.strip().lower()
        if token in {"y", "yes"}:
            return PermissionInteractionResponse(decision="approve")
        if token in {"", "n", "no"}:
            return PermissionInteractionResponse(decision="reject")
        output_func("请输入 y/yes/n/no；空输入默认拒绝。")
        return None

    if isinstance(prompt, QuestionPrompt):
        answer = input.strip()
        if not answer:
            output_func("回答不能为空，请重新输入。")
            return None
        if prompt.options and answer.isdecimal():
            option_index = int(answer) - 1
            if 0 <= option_index < len(prompt.options):
                return QuestionInteractionResponse(answer=prompt.options[option_index])
            output_func("请输入有效的选项编号，或输入自由文本。")
            return None
        return QuestionInteractionResponse(answer=answer)

    raise HITLCheckpointInvalidError("interaction prompt 不受支持")


def _write_result(
    result: RunResult,
    *,
    output_func: Callable[[str], None],
    error_func: Callable[[str], None],
    include_assistant: bool,
) -> None:
    """输出 terminal run 的助手文本与结构化错误。

    Args:
        result (RunResult): terminal run 结果。
        output_func (Callable[[str], None]): 标准输出回调。
        error_func (Callable[[str], None]): 标准错误回调。
        include_assistant (bool): 是否输出 durable assistant 完整文本。
    """
    if include_assistant and result.assistant_message is not None:
        output_func(result.assistant_message.text)
    if result.error is not None:
        error_func(f"{result.error.source}:{result.error.code}: {result.error.message}")


def _format_iris_error(error: IrisError) -> str:
    """把领域异常格式化为稳定运行时错误文本。

    Args:
        error (IrisError): Iris 领域异常。

    Returns:
        str: ``source:code: message`` 格式的文本。
    """
    return f"{error.runtime_source}:{error.runtime_code}: {error.message}"


__all__ = ["ChatOptions", "run_chat", "run_chat_loop"]
