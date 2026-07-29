"""AgentRunner 测试使用的确定性依赖。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.message import LLMRequest, LLMResponse, TextBlock, ToolUseBlock
from iris.runtime import (
    AgentRuntime,
    RuntimeActivationInput,
    RuntimeActivationResult,
    RuntimeCommitPort,
    RuntimeEnvironment,
    RuntimeMessageAssembler,
    ToolBridge,
)
from iris.tools import CancellationSignal, PermissionPolicy, ToolExecutor, ToolRegistry


class FrozenClock:
    """可由测试显式推进的 aware UTC clock。"""

    def __init__(self, now: datetime | None = None) -> None:
        self.current = now or datetime(2026, 1, 2, 3, 4, tzinfo=UTC)

    def now(self) -> datetime:
        """返回当前测试时间。"""
        return self.current

    def advance(self, *, seconds: float) -> None:
        """按秒推进测试时间。"""
        self.current += timedelta(seconds=seconds)


class StaticProvider:
    """按顺序返回固定响应并记录请求的 provider。"""

    def __init__(self, *responses: LLMResponse) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """记录请求并返回下一条固定响应。"""
        self.requests.append(request)
        return self.responses.pop(0)


class BlockingProvider:
    """等待测试显式放行后才返回的 provider。"""

    def __init__(self, response: LLMResponse | None = None) -> None:
        self.response = response or text_response()
        self.requests: list[LLMRequest] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """暴露已进入 provider effect 的同步点。"""
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        return self.response


class CountingAgentRuntime(AgentRuntime):
    """记录每个 activation 调用 ``execute`` 的次数。"""

    def __init__(self, runtime: AgentRuntime) -> None:
        super().__init__(runtime.environment)
        self.execute_calls = 0

    async def execute(
        self,
        activation: RuntimeActivationInput,
        *,
        commits: RuntimeCommitPort,
        cancellation: CancellationSignal,
    ) -> RuntimeActivationResult:
        """记录调用后委托给真实 inner engine。"""
        self.execute_calls += 1
        return await super().execute(
            activation,
            commits=commits,
            cancellation=cancellation,
        )


def text_response(text: str = "完成") -> LLMResponse:
    """构造一个无工具调用的 provider response。"""
    return LLMResponse(
        provider="fake",
        id=f"response-{text}",
        model="fake-model",
        content=[TextBlock(text=text)],
        finish_reason="stop",
        input_tokens=3,
        output_tokens=2,
        total_tokens=5,
    )


def tool_response(call: ToolUseBlock) -> LLMResponse:
    """构造包含一个工具调用的 provider response。"""
    return LLMResponse(
        provider="fake",
        id=f"response-{call.id}",
        model="fake-model",
        content=[TextBlock(text="需要调用工具。"), call],
        finish_reason="tool_calls",
        input_tokens=5,
        output_tokens=3,
        total_tokens=8,
    )


def build_runtime(
    workspace_root: Path,
    *,
    system_text: str = "遵守用户指令",
    registry: ToolRegistry | None = None,
    permission_policy: PermissionPolicy | None = None,
    provider: StaticProvider | None = None,
    agent_name: str = "runner-agent",
) -> AgentRuntime:
    """构造用于 runner 集成测试的真实 ``AgentRuntime``。"""
    resolved_registry = registry or ToolRegistry()
    executor = ToolExecutor(
        resolved_registry,
        permission_policy=permission_policy,
    )
    environment = RuntimeEnvironment(
        agent_config=AgentConfig(
            name=agent_name,
            model={"provider": "openai", "name": "fake-model"},
            system="你是本地助手。",
            permissions={"workspace": ".", "writes": "confirm"},
        ),
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="instructions", content=system_text)])
        ),
        provider=provider or StaticProvider(text_response()),
        assembler=RuntimeMessageAssembler(),
        tool_bridge=ToolBridge(
            tool_view=resolved_registry.view(),
            tool_executor=executor,
        ),
        workspace_root=workspace_root,
    )
    return AgentRuntime(environment)


__all__ = [
    "BlockingProvider",
    "CountingAgentRuntime",
    "FrozenClock",
    "StaticProvider",
    "build_runtime",
    "text_response",
    "tool_response",
]
