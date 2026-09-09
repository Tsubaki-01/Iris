"""Sub Agent 的内部路由、调用契约与模型可见工具。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from ..exceptions import IrisToolExecutionError, IrisToolValidationError
from ..hitl.models import HumanInteraction, SubagentExpiryOwner
from .base import (
    BaseTool,
    CancellationSignal,
    ToolCapability,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
)


@dataclass(frozen=True, slots=True)
class SubagentRoute:
    """已验证的单个 child 配置路由。"""

    selector: str
    config_path: Path
    description: str


@dataclass(frozen=True, slots=True)
class SubagentRouteTable:
    """Catalog 的只读路由快照。"""

    default: str
    routes: Mapping[str, SubagentRoute]


@dataclass(frozen=True, slots=True)
class ResolvedSubagentCall:
    """已解析 prompt 与唯一选中路由。"""

    prompt: str
    route: SubagentRoute


@dataclass(frozen=True, slots=True)
class SubagentParentCall:
    """父运行中原始工具调用的 durable identity。"""

    parent_run_id: str
    parent_tool_call_id: str


@dataclass(frozen=True, slots=True)
class SubagentInvocation:
    """交给 harness 的已验证调用，不复制 lifecycle 状态。"""

    parent_call: SubagentParentCall
    call: ResolvedSubagentCall
    cancellation: CancellationSignal | None


@dataclass(frozen=True, slots=True)
class ChildWaiting:
    """专用执行入口的控制结果，不是模型可见 ToolResult。"""

    child_run_id: str
    child_interaction: HumanInteraction
    proxy_expires_at: datetime | None
    expiry_owner: SubagentExpiryOwner | None


type SubagentExecutionOutcome = ToolResult | ChildWaiting


class SubagentExecutionPort(Protocol):
    """工具层到 harness child 编排的窄接口。"""

    async def execute(self, invocation: SubagentInvocation) -> SubagentExecutionOutcome:
        """执行已解析调用，返回终态结果或人工等待。"""
        ...


class SubagentCallInput(BaseModel):
    """LLM 原始参数的唯一验证边界。"""

    prompt: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    agent: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("agent")
    @classmethod
    def _validate_agent(cls, value: str | None, info: ValidationInfo) -> str | None:
        """只检查 exact membership，selector 不做规范化。"""
        routes: SubagentRouteTable = info.context["routes"]
        if value is not None and value not in routes.routes:
            raise ValueError(f"未知 Sub Agent selector: {value}")
        return value


class SubagentTool(BaseTool):
    """由 ROOT assembly 配置、仅通过专用 executor 执行的工具。"""

    def __init__(self, *, routes: SubagentRouteTable, port: SubagentExecutionPort) -> None:
        self.routes = routes
        self.port = port
        catalog = "\n".join(
            f"- {route.selector}: {route.description}" for route in routes.routes.values()
        )
        self.definition = ToolDefinition(
            name="subagent",
            group="agent",
            capabilities={ToolCapability.AGENT},
            description=(
                "Delegate a focused prompt; wait for completion or human input.\n"
                f"default: {routes.default}\n{catalog}"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "agent": {"type": "string", "enum": list(routes.routes)},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        )

    @property
    def input_model(self) -> type[SubagentCallInput]:
        """返回 raw 调用参数模型。"""
        return SubagentCallInput

    def validate_input(self, params: dict[str, Any]) -> SubagentCallInput:
        """使用本次 catalog 快照解析 LLM 参数。"""
        try:
            return SubagentCallInput.model_validate(params, context={"routes": self.routes})
        except ValidationError as exc:
            raise IrisToolValidationError("Sub Agent 参数校验失败", errors=exc.errors()) from exc

    async def execute_subagent(
        self,
        params: SubagentCallInput,
        context: ToolExecutionContext,
        *,
        parent_call: SubagentParentCall,
    ) -> SubagentExecutionOutcome:
        """投影已验证输入并交给 harness；等待结果只经此入口返回。"""
        selector = self.routes.default if params.agent is None else params.agent
        return await self.port.execute(
            SubagentInvocation(
                parent_call=parent_call,
                call=ResolvedSubagentCall(params.prompt, self.routes.routes[selector]),
                cancellation=context.cancellation,
            )
        )

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        """拒绝绕过完整 child 生命周期的直接调用。"""
        raise IrisToolExecutionError("SubagentTool requires the dedicated executor")
