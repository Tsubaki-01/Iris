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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..exceptions import IrisMemoryError
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
MemoryAccessPolicyFactory = Callable[[ToolExecutionContext], "MemoryAccessPolicy"]


@dataclass(frozen=True, slots=True)
class MemoryAccessPolicy:
    """一次工具执行可使用的记忆访问策略。

    `actor_agent_id` 表示谁在执行；`write_scope` 表示默认写入位置；
    `read_scopes` 表示只读工具允许读取的位置。当前注册的 memory tools 仍然只读，
    但提前把 write/read scope 拆开，便于 runtime 为 subagent 挂载受控策略。
    """

    actor_agent_id: str
    write_scope: MemoryScope
    read_scopes: Sequence[MemoryScope] = ()
    allow_write_shared: bool = False
    allow_promote_to_parent: bool = False

    def effective_write_scope(self, context: ToolExecutionContext) -> MemoryScope:
        """返回本次执行允许写入的默认 scope。"""
        del context
        return self.write_scope

    def effective_read_scopes(self, context: ToolExecutionContext) -> list[MemoryScope]:
        """返回本次执行允许读取的 scope 列表。"""
        del context
        scopes = list(self.read_scopes) or [self.write_scope]
        return _dedupe_scopes(scopes)


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

    def _read_scopes(self, context: ToolExecutionContext) -> list[MemoryScope]:
        """读取当前工具调用允许访问的 read scopes。"""
        return self.access_policy_factory(context).effective_read_scopes(context)


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
        results: list[MemorySearchResult] = []
        for scope in self._read_scopes(context):
            results.extend(
                self.service.recall(
                    MemoryQuery(
                        scope=scope,
                        text=params.query,
                        categories=params.categories,
                        kinds=params.kinds,
                        limit=params.limit,
                    )
                )
            )
        results = _dedupe_results(results)[: params.limit]
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
        categories = [params.category] if params.category is not None else None
        items: list[MemoryItem] = []
        for scope in self._read_scopes(context):
            items.extend(
                self.service.list_items(
                    scope,
                    limit=params.limit,
                    categories=categories,
                )
            )
        items = _dedupe_items(items)[: params.limit]
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
        for scope in self._read_scopes(context):
            item = self.service.get_item(params.item_id, scope)
            if item is not None:
                return self._json_result({"found": True, "item": _item_payload(item)})
        return self._json_result({"found": False})


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


def access_policy_factory_from_scope_factory(
    scope_factory: MemoryScopeFactory,
) -> MemoryAccessPolicyFactory:
    """把旧单 scope factory 包装为默认读写同 scope 的访问策略。"""

    def _factory(context: ToolExecutionContext) -> MemoryAccessPolicy:
        scope = scope_factory(context)
        return MemoryAccessPolicy(
            actor_agent_id=context.agent_id or scope.agent_id,
            write_scope=scope,
            read_scopes=[scope],
        )

    return _factory


def default_memory_access_policy_factory(config: MemoryConfig) -> MemoryAccessPolicyFactory:
    """基于 memory config 构造默认记忆访问策略工厂。"""
    return access_policy_factory_from_scope_factory(default_memory_scope_factory(config))


def register_memory_tools(
    *,
    service: MemoryService,
    scope_factory: MemoryScopeFactory | None = None,
    access_policy_factory: MemoryAccessPolicyFactory | None = None,
    registry: ToolRegistry | None = None,
    max_result_chars: int = 50000,
) -> ToolRegistry:
    """注册只读记忆工具并返回 registry。

    Args:
        service (MemoryService): 供所有记忆工具共享的服务实例。
        scope_factory (MemoryScopeFactory | None): 兼容旧调用方的单 scope 工厂。
            未提供 `access_policy_factory` 时会自动包装成读写同 scope 的访问策略。
        access_policy_factory (MemoryAccessPolicyFactory | None): 基于工具执行上下文生成
            read/write scope 分离访问策略的工厂，推荐由 runtime 为 subagent 挂载。
        registry (ToolRegistry | None): 要扩展的已有 registry。为 None 时创建新 registry。
        max_result_chars (int): 每个记忆工具允许返回给模型的最大字符数。

    Returns:
        ToolRegistry: 已注册 `memory_search`、`memory_list` 和 `memory_get` 的 registry。
            如果传入了 `registry`，返回值就是同一个对象，便于和文件工具等其它工具组合注册。
    """
    if access_policy_factory is None:
        if scope_factory is None:
            raise IrisMemoryError(
                "注册 memory tools 必须提供 scope_factory 或 access_policy_factory"
            )
        access_policy_factory = access_policy_factory_from_scope_factory(scope_factory)
    # 允许调用方把记忆工具追加到已有 registry；未传入时保持独立注册入口的旧行为。
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


def _dedupe_results(results: list[MemorySearchResult]) -> list[MemorySearchResult]:
    """按全局 item id 去重搜索结果，并保持 policy scope 顺序。"""
    seen: set[str] = set()
    deduped: list[MemorySearchResult] = []
    for result in results:
        if result.item.id in seen:
            continue
        seen.add(result.item.id)
        deduped.append(result)
    return deduped


def _dedupe_items(items: list[MemoryItem]) -> list[MemoryItem]:
    """按全局 item id 去重列表结果，并保持 policy scope 顺序。"""
    seen: set[str] = set()
    deduped: list[MemoryItem] = []
    for item in items:
        if item.id in seen:
            continue
        seen.add(item.id)
        deduped.append(item)
    return deduped


def _dedupe_scopes(scopes: list[MemoryScope]) -> list[MemoryScope]:
    """按完整 scope key 去重，避免重复查询同一块记忆。"""
    seen: set[tuple[str, str, str, str, str]] = set()
    deduped: list[MemoryScope] = []
    for scope in scopes:
        key = (
            scope.workspace_id,
            scope.agent_id,
            scope.collection,
            scope.visibility.value,
            scope.session_id or "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(scope)
    return deduped
