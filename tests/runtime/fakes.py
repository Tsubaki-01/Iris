from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from iris.agents import AgentConfig
from iris.context import ContextBuilder, ContextBuildInput
from iris.exceptions import IrisProviderError
from iris.hitl import HumanInteractionService, InMemoryInteractionStore, InteractionStore
from iris.memory import MemoryContextBuilder, MemoryService
from iris.message import LLMRequest, LLMResponse
from iris.runtime import (
    AgentRuntime,
    RuntimeEnvironment,
    RuntimeMessageAssembler,
    RuntimeProvider,
    ToolBridge,
)
from iris.session import InMemorySessionStore, SessionStore
from iris.tools import (
    DefaultPermissionPolicy,
    PermissionPolicy,
    ToolExecutor,
    ToolRegistry,
    ToolRegistryView,
)


class FakeProvider:
    """测试用 provider，只记录请求并按顺序返回预设响应。"""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self._requests: list[LLMRequest] = []

    @property
    def requests(self) -> list[LLMRequest]:
        """返回已捕获的请求快照。"""
        return list(self._requests)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """记录请求并返回下一条预设响应。"""
        self._requests.append(request)
        if not self._responses:
            raise IrisProviderError("FakeProvider 响应已耗尽", provider="fake")
        return self._responses.pop(0)


def build_runtime(
    *,
    agent_config: AgentConfig,
    context_input: ContextBuildInput,
    provider: RuntimeProvider,
    session_store: SessionStore | None = None,
    context_builder: ContextBuilder | None = None,
    assembler: RuntimeMessageAssembler | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_view: ToolRegistryView | None = None,
    tool_executor: ToolExecutor | None = None,
    workspace_root: Path | None = None,
    permission_policy: PermissionPolicy | None = None,
    interaction_store: InteractionStore | None = None,
    interaction_service: HumanInteractionService | None = None,
    memory_service: MemoryService | None = None,
    memory_context_builder: MemoryContextBuilder | None = None,
) -> AgentRuntime:
    """为测试构造包含一致依赖图的 runtime。"""
    resolved_session_store = session_store or InMemorySessionStore()
    registry = tool_registry or (tool_view.registry if tool_view is not None else ToolRegistry())
    resolved_tool_view = tool_view or registry.view()
    resolved_policy = permission_policy or DefaultPermissionPolicy()
    resolved_tool_executor = tool_executor or ToolExecutor(
        registry,
        permission_policy=resolved_policy,
    )
    resolved_interaction_service = interaction_service or HumanInteractionService(
        interaction_store or InMemoryInteractionStore()
    )
    environment = RuntimeEnvironment(
        agent_config=agent_config,
        context_input=context_input,
        provider=provider,
        session_store=resolved_session_store,
        context_builder=context_builder or ContextBuilder(),
        assembler=assembler or RuntimeMessageAssembler(),
        tool_bridge=ToolBridge(
            tool_view=resolved_tool_view,
            tool_executor=resolved_tool_executor,
        ),
        interaction_service=resolved_interaction_service,
        workspace_root=workspace_root or Path.cwd(),
        memory_service=memory_service,
        memory_context_builder=memory_context_builder or MemoryContextBuilder(),
    )
    return AgentRuntime(environment)
