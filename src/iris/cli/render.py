"""Iris CLI 的 Rich 渲染工具。"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ..agents import AgentConfig
from ..exceptions import HITLCheckpointInvalidError
from ..hitl import (
    HumanInteraction,
    InteractionKind,
    PermissionInteractionRequest,
    QuestionInteractionRequest,
)
from ..runtime.models import RuntimeErrorInfo, RuntimeTurnResult
from ..tools import ToolResult
from .trace import TraceStep


class ChatRenderer:
    """渲染 chat CLI 的用户输入、trace、工具结果和助手回复。"""

    def __init__(self, console: Console | None = None) -> None:
        """初始化 renderer。"""
        self.console = console or Console()

    def render_header(
        self,
        *,
        agent_config: AgentConfig,
        session_id: str,
        workspace: str,
        trace_mode: str,
    ) -> None:
        """渲染启动信息。"""
        tool_names = ", ".join(agent_config.tools.builtin) or "无"
        body = (
            f"agent: {agent_config.name}\n"
            f"model: {agent_config.model.provider}/{agent_config.model.name}\n"
            f"session: {session_id}\n"
            f"workspace: {workspace}\n"
            f"tools: {tool_names}\n"
            f"trace: {trace_mode}\n"
            f"writes: {agent_config.permissions.writes}"
        )
        self.console.print(Panel(body, title="Iris Chat", border_style="cyan"))

    def render_help(self, trace_mode: str) -> None:
        """渲染交互命令帮助。"""
        body = (
            "/help 显示帮助\n"
            "/trace off|compact|full 切换 trace 展示\n"
            "/exit 或 /quit 退出\n"
            f"当前 trace: {trace_mode}"
        )
        self.console.print(Panel(body, title="HELP", border_style="blue"))

    def render_user_turn(self, turn_index: int, text: str) -> None:
        """渲染用户输入。"""
        self.console.print(Panel(text, title=f"USER #{turn_index}", border_style="blue"))

    def render_permission_interaction(self, interaction: HumanInteraction) -> None:
        """渲染一次精确工具调用的权限确认。"""
        request = interaction.request
        if interaction.kind is not InteractionKind.PERMISSION or not isinstance(
            request, PermissionInteractionRequest
        ):
            raise HITLCheckpointInvalidError("permission interaction kind/request 不匹配")
        arguments = json.dumps(request.arguments, ensure_ascii=False, indent=2)
        body = (
            f"interaction_id: {interaction.interaction_id}\n"
            f"tool_name: {request.tool_name}\n"
            f"arguments:\n{arguments}\n"
            f"reason: {request.reason}\n"
            f"workspace_root: {request.workspace_root}\n"
            "本次批准只适用于该调用。"
        )
        self.console.print(Panel(body, title="PERMISSION [y/N]", border_style="yellow"))

    def render_question_interaction(self, interaction: HumanInteraction) -> None:
        """渲染人工问题及可选答案。"""
        request = interaction.request
        if interaction.kind is not InteractionKind.QUESTION or not isinstance(
            request, QuestionInteractionRequest
        ):
            raise HITLCheckpointInvalidError("question interaction kind/request 不匹配")
        options = "\n".join(
            f"{index}. {option}" for index, option in enumerate(request.options, start=1)
        )
        option_block = f"{options}\n" if options else ""
        body = (
            f"interaction_id: {interaction.interaction_id}\n"
            f"question: {request.question}\n"
            f"{option_block}"
            "也可输入自由文本。"
        )
        self.console.print(Panel(body, title="QUESTION", border_style="yellow"))

    def render_recovery_notice(self, interaction: HumanInteraction) -> None:
        """渲染 startup recovery 提示。"""
        body = (
            "正在恢复未完成的人工交互。\n"
            f"interaction_id: {interaction.interaction_id}\n"
            f"kind: {interaction.kind.value}\n"
            f"status: {interaction.status.value}"
        )
        self.console.print(Panel(body, title="RECOVERY", border_style="yellow"))

    def render_trace_compact(self, steps: list[TraceStep]) -> None:
        """渲染简洁 trace 表。"""
        for step in steps:
            self.console.print(
                Panel(
                    _request_summary(step),
                    title=f"REQUEST {step.turn_index}.{step.step_index}",
                    border_style="white",
                )
            )
            if step.response is not None:
                self.console.print(
                    Panel(
                        _response_summary(step),
                        title=f"RESPONSE {step.turn_index}.{step.step_index}",
                        border_style="green",
                    )
                )
            elif step.error:
                self.console.print(
                    Panel(
                        step.error,
                        title=f"ERROR {step.turn_index}.{step.step_index}",
                        border_style="red",
                    )
                )

    def render_trace_full(self, steps: list[TraceStep]) -> None:
        """渲染完整 JSON trace。"""
        for step in steps:
            payload = json.dumps(step.snapshot(), ensure_ascii=False, indent=2)
            self.console.print(
                Panel(
                    Syntax(payload, "json", word_wrap=True),
                    title=f"TRACE {step.turn_index}.{step.step_index}",
                    border_style="white",
                )
            )

    def render_tool_results(self, result: RuntimeTurnResult) -> None:
        """渲染工具执行结果。"""
        if not result.tool_results:
            return
        table = Table(title="TOOL RESULTS", show_lines=False)
        table.add_column("tool", style="cyan")
        table.add_column("status")
        table.add_column("preview")
        for tool_result in result.tool_results:
            table.add_row(
                tool_result.tool_name,
                "error" if tool_result.is_error else "ok",
                _tool_preview(tool_result),
            )
        self.console.print(table)

    def render_assistant(self, result: RuntimeTurnResult) -> None:
        """渲染助手最终回复或 runtime 状态。"""
        if result.error is not None:
            self.render_error(result.error)
        if result.assistant_message is None:
            return
        text = result.assistant_message.text or _assistant_tool_call_summary(result)
        self.console.print(Panel(text, title="ASSISTANT", border_style="green"))

    def render_error(self, error: RuntimeErrorInfo | Exception) -> None:
        """渲染结构化错误。"""
        if isinstance(error, RuntimeErrorInfo):
            text = f"{error.source}:{error.code}\n{error.message}"
        else:
            text = f"{error.__class__.__name__}\n{error}"
        self.console.print(Panel(text, title="ERROR", border_style="red"))

    def render_warning(self, message: str) -> None:
        """渲染警告。"""
        self.console.print(Panel(message, title="WARNING", border_style="yellow"))


def _request_summary(step: TraceStep) -> str:
    """生成 request 摘要。"""
    request = step.request
    roles = " -> ".join(message.role.value for message in request.messages)
    tool_names = _tool_schema_names(request.tools)
    latest_preview = _latest_message_preview(request.messages)
    return (
        f"model: {request.model}\n"
        f"messages: {len(request.messages)}\n"
        f"roles: {roles}\n"
        f"tools: {', '.join(tool_names) if tool_names else '无'}\n"
        f"tool_choice: {request.tool_choice or 'default'}\n"
        f"latest: {latest_preview}"
    )


def _response_summary(step: TraceStep) -> str:
    """生成 response 摘要。"""
    response = step.response
    if response is None:
        return "无 response"
    tool_calls = [block.name for block in response.to_msg().tool_calls]
    text = response.to_msg().text[:500]
    return (
        f"provider: {response.provider}\n"
        f"model: {response.model}\n"
        f"finish_reason: {response.finish_reason}\n"
        f"usage: {response.input_tokens}/{response.output_tokens}/{response.total_tokens}\n"
        f"tool_calls: {', '.join(tool_calls) if tool_calls else '无'}\n"
        f"text: {text}"
    )


def _tool_schema_names(tools: list[dict[str, Any]]) -> list[str]:
    """提取工具 schema 名称。"""
    names: list[str] = []
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
        elif isinstance(tool.get("name"), str):
            names.append(tool["name"])
    return names


def _latest_message_preview(messages: list[Any]) -> str:
    """返回最后一条消息的摘要。"""
    if not messages:
        return ""
    message = messages[-1]
    text = message.text
    if text:
        return text[:500]
    if message.tool_results:
        return "\n".join(result.content for result in message.tool_results)[:500]
    if message.tool_calls:
        return ", ".join(call.name for call in message.tool_calls)
    return ""


def _tool_preview(tool_result: ToolResult) -> str:
    """返回工具结果摘要。"""
    text = tool_result.model_content
    return text[:500]


def _assistant_tool_call_summary(result: RuntimeTurnResult) -> str:
    """在 assistant 只有工具调用时返回摘要。"""
    if result.assistant_message is None:
        return ""
    calls = [call.name for call in result.assistant_message.tool_calls]
    if not calls:
        return ""
    return "工具调用: " + ", ".join(calls)


__all__ = ["ChatRenderer"]
