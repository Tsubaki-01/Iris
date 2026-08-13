"""基于 ``AgentRunner`` 的标准库交互式 chat CLI。

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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import init_config, is_config_initialized
from ..exceptions import HITLCheckpointInvalidError, IrisError
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
    """装配 complete-run harness 并启动 chat。

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
            init_config(
                env_file=str(options.env_file) if options.env_file is not None else None
            )
        runner = AgentRunner.from_config_path(options.config_path)
    except IrisError as exc:
        write_error(_format_iris_error(exc))
        return 1

    return run_chat_loop(
        runner=runner,
        options=options,
        input_func=input_func,
        output_func=output_func,
        error_func=write_error,
    )


def run_chat_loop(
    *,
    runner: AgentRunner,
    options: ChatOptions,
    input_func: Callable[[str], str] | None = None,
    output_func: Callable[[str], None] | None = None,
    error_func: Callable[[str], None] | None = None,
) -> int:
    """在单个 event loop 中执行可测试的同步 chat host。

    Args:
        runner (AgentRunner): complete-run SDK facade。
        options (ChatOptions): chat 命令选项。
        input_func (Callable[[str], str] | None): 可选输入回调。
        output_func (Callable[[str], None] | None): 可选标准输出回调。
        error_func (Callable[[str], None] | None): 可选标准错误回调。

    Returns:
        int: 进程退出码。
    """
    read_input = input_func or builtins.input
    write_output = output_func or builtins.print
    write_error = error_func or (lambda message: print(message, file=sys.stderr))
    return asyncio.run(
        _run_chat_loop_async(
            runner=runner,
            options=options,
            input_func=read_input,
            output_func=write_output,
            error_func=write_error,
        )
    )


async def _run_chat_loop_async(
    *,
    runner: AgentRunner,
    options: ChatOptions,
    input_func: Callable[[str], str],
    output_func: Callable[[str], None],
    error_func: Callable[[str], None],
) -> int:
    """依次创建并推进 chat 中的 logical runs。

    Args:
        runner (AgentRunner): complete-run SDK facade。
        options (ChatOptions): chat 命令选项。
        input_func (Callable[[str], str]): 已解析的输入回调。
        output_func (Callable[[str], None]): 已解析的标准输出回调。
        error_func (Callable[[str], None]): 已解析的标准错误回调。

    Returns:
        int: 进程退出码。
    """
    while True:
        try:
            user_input = input_func("iris> ")
        except KeyboardInterrupt:
            return 130
        except EOFError:
            return 0

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            output_func("可用命令：")
            output_func("/help  显示帮助")
            output_func("/exit  退出 chat")
            output_func("/quit  退出 chat")
            continue
        if user_input.startswith("/"):
            output_func("未知命令。输入 /help 查看可用命令。")
            continue

        try:
            result = await runner.start(
                AgentRunRequest(input=user_input, session_id=options.session_id),
                options=AgentRunOptions(
                    limits=RunLimits(max_model_steps=options.max_steps),
                    runtime=RuntimeExecutionOptions(include_tools=options.include_tools),
                ),
            )
            result = await _resume_until_terminal(
                runner,
                result,
                input_func=input_func,
                output_func=output_func,
            )
        except KeyboardInterrupt:
            return 130
        except EOFError:
            return 0
        except IrisError as exc:
            error_func(_format_iris_error(exc))
            return 1

        _write_result(result, output_func=output_func, error_func=error_func)
        if result.run.stop_reason in {
            RunStopReason.FAILED,
            RunStopReason.OUTCOME_UNKNOWN,
        }:
            return 1


async def _resume_until_terminal(
    runner: AgentRunner,
    result: RunResult,
    *,
    input_func: Callable[[str], str],
    output_func: Callable[[str], None],
) -> RunResult:
    """依次处理 logical run 的人工交互。

    Args:
        runner (AgentRunner): complete-run SDK facade。
        result (RunResult): 当前 waiting 或 terminal 结果。
        input_func (Callable[[str], str]): 输入回调。
        output_func (Callable[[str], None]): 标准输出回调。

    Returns:
        RunResult: 最终的 terminal 结果。

    Raises:
        HITLCheckpointInvalidError: waiting 结果缺少交互或提示类型不受支持。
    """
    while result.run.phase is RunPhase.WAITING:
        interaction = result.pending_interaction
        if interaction is None:
            raise HITLCheckpointInvalidError("waiting 结果缺少 pending interaction")
        response = _collect_interaction_response(
            interaction,
            input_func=input_func,
            output_func=output_func,
        )
        result = await runner.resume(
            result.run.run_id,
            interaction_id=interaction.interaction_id,
            response=response,
        )
    return result


def _collect_interaction_response(
    interaction: HumanInteraction,
    *,
    input_func: Callable[[str], str],
    output_func: Callable[[str], None],
) -> PermissionInteractionResponse | QuestionInteractionResponse:
    """把终端输入映射为 typed HITL response。

    Args:
        interaction (HumanInteraction): 当前 pending interaction。
        input_func (Callable[[str], str]): 输入回调。
        output_func (Callable[[str], None]): 标准输出回调。

    Returns:
        PermissionInteractionResponse | QuestionInteractionResponse:
            与 prompt 类型匹配的响应。

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
        while True:
            token = input_func("批准该调用？ [y/N] ").strip().lower()
            if token in {"y", "yes"}:
                return PermissionInteractionResponse(decision="approve")
            if token in {"", "n", "no"}:
                return PermissionInteractionResponse(decision="reject")
            output_func("请输入 y/yes/n/no；空输入默认拒绝。")

    if isinstance(prompt, QuestionPrompt):
        output_func(prompt.question)
        for index, option in enumerate(prompt.options, start=1):
            output_func(f"{index}. {option}")
        while True:
            answer = input_func("回答> ").strip()
            if not answer:
                output_func("回答不能为空，请重新输入。")
                continue
            if prompt.options and answer.isdecimal():
                option_index = int(answer) - 1
                if 0 <= option_index < len(prompt.options):
                    return QuestionInteractionResponse(answer=prompt.options[option_index])
                output_func("请输入有效的选项编号，或输入自由文本。")
                continue
            return QuestionInteractionResponse(answer=answer)

    raise HITLCheckpointInvalidError("interaction prompt 不受支持")


def _write_result(
    result: RunResult,
    *,
    output_func: Callable[[str], None],
    error_func: Callable[[str], None],
) -> None:
    """输出 terminal run 的助手文本与结构化错误。

    Args:
        result (RunResult): terminal run 结果。
        output_func (Callable[[str], None]): 标准输出回调。
        error_func (Callable[[str], None]): 标准错误回调。
    """
    if result.assistant_message is not None:
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
