"""记忆只读工具。

本模块只提供 `memory_search`、`memory_list`、`memory_get` 三个只读工具。
写入和删除能力仍由 Python SDK 暴露，不在 Stage 4 默认注册为工具。

Example:
    registry = register_memory_tools(service=service, scope_factory=factory)
"""

# region imports
from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..message import TextBlock
from ..tools import (
    BaseTool,
    ToolCapability,
    ToolDefinition,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
    schema_from_pydantic_model,
)
from .config import MemoryConfig
from .models import (
    MemoryCategory,
    MemoryItem,
    MemoryItemKind,
    MemoryQuery,
    MemoryScope,
    MemorySearchResult,
)
from .service import MemoryService

# endregion

InputT = TypeVar("InputT", bound=BaseModel)
MemoryScopeFactory = Callable[[ToolExecutionContext], MemoryScope]


class MemorySearchToolInput(BaseModel):
    """记忆搜索工具输入。"""

    model_config = ConfigDict(extra="forbid")

    query: str
    limit: int = 8
    categories: list[MemoryCategory] = Field(default_factory=list)
    kinds: list[MemoryItemKind] = Field(default_factory=list)

    @field_validator("limit")
    @classmethod
    def _validate_limit(cls, value: int) -> int:
        """限制搜索返回数量。"""
        if value <= 0:
            raise ValueError("limit 必须为正数")
        return min(value, 100)


class MemoryListToolInput(BaseModel):
    """记忆列表工具输入。"""

    model_config = ConfigDict(extra="forbid")

    limit: int = 50
    category: MemoryCategory | None = None

    @field_validator("limit")
    @classmethod
    def _validate_limit(cls, value: int) -> int:
        """限制列表返回数量。"""
        if value <= 0:
            raise ValueError("limit 必须为正数")
        return min(value, 100)


class MemoryGetToolInput(BaseModel):
    """记忆读取工具输入。"""

    model_config = ConfigDict(extra="forbid")

    item_id: str

    @field_validator("item_id")
    @classmethod
    def _validate_item_id(cls, value: str) -> str:
        """校验 item id 不能为空。"""
        if not value.strip():
            raise ValueError("item_id 不能为空")
        return value


class MemoryTool(BaseTool, Generic[InputT]):  # noqa: UP046
    """记忆工具协议适配基类。"""

    name: ClassVar[str]
    description: ClassVar[str]
    input_type: type[InputT]
    capabilities: ClassVar[set[ToolCapability]] = {ToolCapability.READ}

    def __init__(
        self,
        *,
        service: MemoryService,
        scope_factory: MemoryScopeFactory,
        max_result_chars: int = 50000,
    ) -> None:
        """创建记忆工具实例。"""
        self.service = service
        self.scope_factory = scope_factory
        self.definition = ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=schema_from_pydantic_model(self.input_type),
            capabilities=self.capabilities,
            group="memory",
            max_result_chars=max_result_chars,
        )

    @property
    def input_model(self) -> type[BaseModel] | None:
        """返回工具输入模型。"""
        return self.input_type

    def validate_input(self, params: dict[str, Any]) -> BaseModel:
        """校验原始工具调用参数。"""
        return self.input_type.model_validate(params)

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """适配工具协议并调用具体记忆业务。"""
        input_data = cast(InputT, self.input_type.model_validate(params))
        return await self._impl(input_data, context)

    @abstractmethod
    async def _impl(self, params: InputT, context: ToolExecutionContext) -> ToolResult:
        """执行具体只读记忆工具。"""
        raise NotImplementedError

    def _json_result(self, payload: dict[str, Any]) -> ToolResult:
        """构造 JSON 文本工具结果。"""
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return ToolResult(
            tool_use_id="",
            tool_name=self.name,
            content=[TextBlock(text=content)],
        )


class MemorySearchTool(MemoryTool[MemorySearchToolInput]):
    """搜索当前 scope 内的长期记忆。"""

    name: ClassVar[str] = "memory_search"
    description: ClassVar[str] = "搜索当前 agent scope 内的长期记忆"
    input_type: type[MemorySearchToolInput] = MemorySearchToolInput

    async def _impl(
        self,
        params: MemorySearchToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """调用 MemoryService.recall 执行搜索。"""
        scope = self.scope_factory(context)
        results = self.service.recall(
            MemoryQuery(
                scope=scope,
                text=params.query,
                categories=params.categories,
                kinds=params.kinds,
                limit=params.limit,
            )
        )
        return self._json_result({"results": [_result_payload(result) for result in results]})


class MemoryListTool(MemoryTool[MemoryListToolInput]):
    """列出当前 scope 内的长期记忆。"""

    name: ClassVar[str] = "memory_list"
    description: ClassVar[str] = "列出当前 agent scope 内的长期记忆"
    input_type: type[MemoryListToolInput] = MemoryListToolInput

    async def _impl(
        self,
        params: MemoryListToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """调用 MemoryService.list_items 执行列表读取。"""
        scope = self.scope_factory(context)
        items = self.service.list_items(scope, limit=params.limit)
        if params.category is not None:
            items = [item for item in items if item.category == params.category]
        return self._json_result({"items": [_item_payload(item) for item in items]})


class MemoryGetTool(MemoryTool[MemoryGetToolInput]):
    """读取当前 scope 内的一条长期记忆。"""

    name: ClassVar[str] = "memory_get"
    description: ClassVar[str] = "按 id 读取当前 agent scope 内的一条长期记忆"
    input_type: type[MemoryGetToolInput] = MemoryGetToolInput

    async def _impl(
        self,
        params: MemoryGetToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """调用 MemoryService.get_item 读取单条记忆。"""
        item = self.service.get_item(params.item_id, self.scope_factory(context))
        if item is None:
            return self._json_result({"found": False})
        return self._json_result({"found": True, "item": _item_payload(item)})


MEMORY_TOOL_CLASSES: tuple[type[MemoryTool[Any]], ...] = (
    MemorySearchTool,
    MemoryListTool,
    MemoryGetTool,
)


def default_memory_scope_factory(config: MemoryConfig) -> MemoryScopeFactory:
    """基于 memory config 和工具上下文构造 scope factory。"""

    def _factory(context: ToolExecutionContext) -> MemoryScope:
        return config.scope.to_scope(
            workspace_id=str(context.workspace_root.resolve(strict=False)),
            agent_id=context.agent_id or "default",
            session_id=context.session_id or None,
        )

    return _factory


def register_memory_tools(
    *,
    service: MemoryService,
    scope_factory: MemoryScopeFactory,
    registry: ToolRegistry | None = None,
    max_result_chars: int = 50000,
) -> ToolRegistry:
    """注册只读记忆工具并返回 registry。

    Args:
        service (MemoryService): 供所有记忆工具共享的服务实例。
        scope_factory (MemoryScopeFactory): 基于工具执行上下文生成记忆 scope 的工厂。
        registry (ToolRegistry | None): 要扩展的已有 registry。为 None 时创建新 registry。
        max_result_chars (int): 每个记忆工具允许返回给模型的最大字符数。

    Returns:
        ToolRegistry: 已注册 `memory_search`、`memory_list` 和 `memory_get` 的 registry。
            如果传入了 `registry`，返回值就是同一个对象，便于和文件工具等其它工具组合注册。
    """
    # 允许调用方把记忆工具追加到已有 registry；未传入时保持独立注册入口的旧行为。
    registry = registry or ToolRegistry()
    for tool_cls in MEMORY_TOOL_CLASSES:
        registry.register(
            tool_cls(
                service=service,
                scope_factory=scope_factory,
                max_result_chars=max_result_chars,
            )
        )
    return registry


def _item_payload(item: MemoryItem) -> dict[str, Any]:
    """转换长期记忆条目为工具输出 payload。"""
    payload: dict[str, Any] = {
        "id": item.id,
        "text": item.text,
        "category": item.category.value,
        "kind": item.kind.value,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }
    if item.confidence is not None:
        payload["confidence"] = item.confidence
    if item.importance is not None:
        payload["importance"] = item.importance
    return payload


def _result_payload(result: MemorySearchResult) -> dict[str, Any]:
    """转换搜索结果为工具输出 payload。"""
    payload = _item_payload(result.item)
    payload["score"] = result.score
    payload["source"] = result.source
    return payload
