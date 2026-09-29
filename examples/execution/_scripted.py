"""示例共享的离线 provider 与显式 permission response。"""

from collections import deque
from pathlib import Path
from typing import Any

from iris.agents import AgentConfig, load_agent_config
from iris.exceptions import IrisProviderError, IrisRunStateError, IrisToolExecutionError
from iris.harness import AgentRunner
from iris.hitl import PermissionInteractionResponse, PermissionPrompt
from iris.lifecycle import RunPhase, RunResult
from iris.message import LLMRequest, LLMResponse, TextBlock, ToolUseBlock


class ScriptedProvider:
    """仅返回宿主预先声明的响应，不创建网络 client。"""

    def __init__(self) -> None:
        self.steps: deque[LLMResponse | IrisProviderError] = deque()

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """为小型固定脚本提供计量，不调用外部 tokenizer。"""
        return 1

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """取下一条已声明响应或模拟 provider 错误。"""
        step = self.steps.popleft()
        if isinstance(step, IrisProviderError):
            raise step
        return step


def tool(call_id: str, name: str, **arguments: Any) -> LLMResponse:
    """构造单个真实工具调用，交给正常 runtime 执行。"""
    return LLMResponse(
        provider="scripted",
        model="offline-example",
        content=[ToolUseBlock(id=call_id, name=name, input=arguments)],
        finish_reason="tool_calls",
    )


def done(text: str) -> LLMResponse:
    """构造示例的最终回答。"""
    return LLMResponse(
        provider="scripted",
        model="offline-example",
        content=[TextBlock(text=text)],
        finish_reason="stop",
    )


def config_for_workspace(name: str, workspace: Path) -> AgentConfig:
    """加载完整示例 YAML，仅替换本次明确指定的 workspace。"""
    config = load_agent_config(Path(__file__).parent / name)
    return config.model_copy(
        update={"permissions": config.permissions.model_copy(update={"workspace": str(workspace)})}
    )


async def approve_commands(
    runner: AgentRunner, result: RunResult, confirmed: list[str]
) -> RunResult:
    """示例宿主显式批准固定脚本；普通 question 留给调用者处理。"""
    while result.run.phase is RunPhase.WAITING:
        interaction = result.pending_interaction
        assert interaction is not None
        if not isinstance(interaction.request.prompt, PermissionPrompt):
            break
        confirmed.append(interaction.request.tool_call.tool_name)
        result = await runner.resume(
            result.run.run_id,
            interaction_id=interaction.interaction_id,
            response=PermissionInteractionResponse(decision="approve"),
        )
    return result


def require_completed(runner: AgentRunner, result: RunResult) -> None:
    """固定 provider 不会推理工具错误，因此由示例宿主检查真实结果。"""
    if result.run.stop_reason is None or result.run.stop_reason.value != "completed":
        raise IrisRunStateError(f"示例没有完成: {result.error}", run_id=result.run.run_id)
    for call in runner.list_tool_calls(result.run.run_id):
        if call.result is not None and call.result.is_error:
            raise IrisToolExecutionError(call.result.model_content, tool_name=call.tool_name)
