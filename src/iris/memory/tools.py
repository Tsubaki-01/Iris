"""记忆只读工具。

本模块只提供 `memory_search`、`memory_list`、`memory_get` 三个只读工具。
写入和删除能力仍由 Python SDK 暴露，不在 Stage 4 默认注册为工具。

Example:
    registry = register_memory_tools(
        service=service,
        access_policy_factory=policy_factory,
    )
"""

# region imports
from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
    MemorySearchResult,
)
from .service import MemoryService

# endregion

InputT = TypeVar("InputT", bound=BaseModel)
MemoryAccessPolicyFactory = Callable[[ToolExecutionContext], "MemoryAccessPolicy"]


@dataclass(frozen=True, slots=True)
class MemoryAccessPolicy:
    """一次工具执行可使用的记忆访问策略。

    读写 namespace 由宿主绑定；空读取集合表示不读取任何空间。
    """

    read_namespaces: Sequence[str] = ("project",)
    write_namespace: str = "project"


class MemorySearchToolInput(BaseModel):
    """记忆搜索工具输入。"""

    model_config = ConfigDict(extra="forbid")

    query: str
    limit: int = Field(default=8, gt=0, le=100)
    categories: list[MemoryCategory] = Field(default_factory=list)
    kinds: list[MemoryItemKind] = Field(default_factory=list)


class MemoryListToolInput(BaseModel):
    """记忆列表工具输入。"""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=50, gt=0, le=100)
    category: MemoryCategory | None = None


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
        access_policy_factory: MemoryAccessPolicyFactory,
        max_result_chars: int = 50000,
    ) -> None:
        """创建记忆工具实例。"""
        self.service = service
        self.access_policy_factory = access_policy_factory
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
        input_data = cast(InputT, params)
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

    def _read_namespaces(self, context: ToolExecutionContext) -> list[str]:
        """取得宿主为当前调用绑定的读取范围。"""
        return list(self.access_policy_factory(context).read_namespaces)


class MemorySearchTool(MemoryTool[MemorySearchToolInput]):
    """联合搜索允许读取的 namespace。"""

    name: ClassVar[str] = "memory_search"
    description: ClassVar[str] = "搜索允许读取的项目记忆"
    input_type: type[MemorySearchToolInput] = MemorySearchToolInput

    async def _impl(
        self,
        params: MemorySearchToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """调用 MemoryService.recall 执行搜索。"""
        results = await self.service.arecall(
            MemoryQuery.model_construct(
                namespaces=self._read_namespaces(context),
                text=params.query,
                categories=params.categories,
                kinds=params.kinds,
                limit=params.limit,
            )
        )
        return self._json_result({"results": [_result_payload(result) for result in results]})


class MemoryListTool(MemoryTool[MemoryListToolInput]):
    """联合列出允许读取的 namespace 内的长期记忆。"""

    name: ClassVar[str] = "memory_list"
    description: ClassVar[str] = "列出允许读取的项目记忆"
    input_type: type[MemoryListToolInput] = MemoryListToolInput

    async def _impl(
        self,
        params: MemoryListToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """调用 MemoryService.list_items 执行列表读取。"""
        categories = [params.category] if params.category is not None else None
        items = await self.service.alist_items(
            self._read_namespaces(context), limit=params.limit, categories=categories
        )
        return self._json_result({"items": [_item_payload(item) for item in items]})


class MemoryGetTool(MemoryTool[MemoryGetToolInput]):
    """按 ID 在允许读取的 namespace 中定位记忆。"""

    name: ClassVar[str] = "memory_get"
    description: ClassVar[str] = "按 id 读取允许范围内的一条项目记忆"
    input_type: type[MemoryGetToolInput] = MemoryGetToolInput

    async def _impl(
        self,
        params: MemoryGetToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """调用 MemoryService.get_item 读取单条记忆。"""
        item = await self.service.aget_item(params.item_id, self._read_namespaces(context))
        if item is not None:
            return self._json_result({"found": True, "item": _item_payload(item)})
        return self._json_result({"found": False})


MEMORY_TOOL_CLASSES: tuple[type[MemoryTool[Any]], ...] = (
    MemorySearchTool,
    MemoryListTool,
    MemoryGetTool,
)


def default_memory_access_policy_factory(
    config: MemoryConfig,
) -> MemoryAccessPolicyFactory:
    """基于 memory config 构造默认记忆访问策略工厂。"""

    def _factory(context: ToolExecutionContext) -> MemoryAccessPolicy:
        return MemoryAccessPolicy(
            read_namespaces=config.read_namespaces,
            write_namespace=config.write_namespace,
        )

    return _factory


def register_memory_tools(
    *,
    service: MemoryService,
    access_policy_factory: MemoryAccessPolicyFactory,
    registry: ToolRegistry | None = None,
    max_result_chars: int = 50000,
) -> ToolRegistry:
    """注册只读记忆工具并返回 registry。

    Args:
        service (MemoryService): 供所有记忆工具共享的服务实例。
        access_policy_factory (MemoryAccessPolicyFactory): 基于工具执行上下文生成
            read/write namespace 分离访问策略的工厂。
        registry (ToolRegistry | None): 要扩展的已有 registry。为 None 时创建新 registry。
        max_result_chars (int): 每个记忆工具允许返回给模型的最大字符数。

    Returns:
        ToolRegistry: 已注册 `memory_search`、`memory_list` 和 `memory_get` 的 registry。
            如果传入了 `registry`，返回值就是同一个对象，便于和文件工具等其它工具组合注册。
    """
    registry = registry or ToolRegistry()
    for tool_cls in MEMORY_TOOL_CLASSES:
        registry.register(
            tool_cls(
                service=service,
                access_policy_factory=access_policy_factory,
                max_result_chars=max_result_chars,
            )
        )
    return registry


def _item_payload(item: MemoryItem) -> dict[str, Any]:
    """转换长期记忆条目为工具输出 payload。"""
    payload: dict[str, Any] = {
        "id": item.id,
        "namespace": item.namespace,
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
