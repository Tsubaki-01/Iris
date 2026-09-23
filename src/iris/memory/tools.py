"""项目记忆的读写工具。

Search/Fetch 与写工具均由宿主显式声明；默认不注册任何工具。

Example:
    registry = register_memory_tools(
        service=service,
        access_policy_factory=policy_factory,
        tool_names=("memory.search", "memory.fetch"),
    )
"""

# region imports
from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

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
    MemoryActor,
    MemoryCategory,
    MemoryItem,
    MemoryItemKind,
    MemoryItemPatch,
    MemorySearchHit,
    MemorySearchQuery,
    MemorySourceType,
    MemoryWriteInput,
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


class MemoryFetchToolInput(BaseModel):
    """按已知条目 ID 获取允许范围内的当前完整记录。"""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(pattern=r"\S")


class MemoryRememberToolInput(BaseModel):
    """创建记忆的业务输入，写入范围和来源由宿主绑定。"""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(pattern=r"\S")
    reason: str = Field(pattern=r"\S")
    category: MemoryCategory = MemoryCategory.USER
    kind: MemoryItemKind = MemoryItemKind.NOTE


class MemoryUpdateToolInput(BaseModel):
    """修改同一记忆条目的业务输入。"""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(pattern=r"\S")
    patch: MemoryItemPatch
    reason: str = Field(pattern=r"\S")


class MemoryForgetToolInput(BaseModel):
    """软删除记忆的业务输入。"""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(pattern=r"\S")
    reason: str = Field(pattern=r"\S")


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
        """执行具体记忆工具。"""
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

    async def _write_result(self, namespace: str, payload: dict[str, Any]) -> ToolResult:
        """数据库提交后把正文未同步状态同时交给模型。"""
        warning = await self.service.run_async_io(
            lambda: self.service.projection_warning(namespace)
        )
        if warning:
            payload["warning"] = warning
        return self._json_result(payload)


class MemorySearchTool(MemoryTool[MemorySearchQuery]):
    """联合搜索允许读取的 namespace，返回可直接使用的原文片段。"""

    name: ClassVar[str] = "memory_search"
    description: ClassVar[str] = (
        "搜索允许读取的记忆，返回条目 ID、原文片段及完整性标记。"
        "query 拆词后按 OR 匹配，加词不保证收紧。"
        "可选 required_terms 指定有依据的必要原文词组，各词组与 query 同时满足。"
        "词组按现有分词进行有序相邻匹配，非逐字匹配；英文忽略大小写。"
        "categories/kinds 仅在已知存储标签时填写。"
        "空结果只表示本次查询未命中，可根据证据调整条件。"
    )
    input_type: type[MemorySearchQuery] = MemorySearchQuery

    async def _impl(self, params: MemorySearchQuery, context: ToolExecutionContext) -> ToolResult:
        """直接传递已验证查询和本次宿主读取范围。"""
        response = await self.service.asearch(params, self._read_namespaces(context))
        payload: dict[str, Any] = {
            "items": [_hit_payload(hit) for hit in response.items],
            "has_more": response.has_more,
        }
        if response.has_more:
            payload["hint"] = "还有候选；这不要求继续查询。"
        return self._json_result(payload)


class MemoryFetchTool(MemoryTool[MemoryFetchToolInput]):
    """按 ID 获取允许读取范围内的当前活跃记录。"""

    name: ClassVar[str] = "memory_fetch"
    description: ClassVar[str] = "按 item_id 获取一条当前记忆的完整正文和元数据"
    input_type: type[MemoryFetchToolInput] = MemoryFetchToolInput

    async def _impl(
        self, params: MemoryFetchToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        """复用当前记录读取，并将缺失、非活跃和范围外统一报告为读取错误。"""
        item = await self.service.aget_item(params.item_id, self._read_namespaces(context))
        if item is None:
            raise IrisMemoryError("允许读取范围内未找到有效记忆", item_id=params.item_id)
        return self._json_result({"item": item.model_dump(mode="json")})


class MemoryRememberTool(MemoryTool[MemoryRememberToolInput]):
    """在宿主绑定的 namespace 中创建长期记忆。"""

    name: ClassVar[str] = "memory_remember"
    description: ClassVar[str] = "保存需要后续复用的项目记忆，并说明保存原因"
    input_type: type[MemoryRememberToolInput] = MemoryRememberToolInput
    capabilities: ClassVar[set[ToolCapability]] = {ToolCapability.WRITE}

    async def _impl(
        self, params: MemoryRememberToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        """把已校验业务字段投影为写入请求，来源固定为当前工具调用。"""
        item = await self.service.aremember(
            MemoryWriteInput.model_construct(
                namespace=self.access_policy_factory(context).write_namespace,
                text=params.text,
                reason=params.reason,
                category=params.category,
                kind=params.kind,
                actor=MemoryActor.AGENT,
                source_type=MemorySourceType.TOOL_EVENT,
                source_id=context.call_id,
            )
        )
        return await self._write_result(item.namespace, {"item": _item_payload(item)})


class MemoryUpdateTool(MemoryTool[MemoryUpdateToolInput]):
    """只更新宿主绑定的 namespace 中的条目。"""

    name: ClassVar[str] = "memory_update"
    description: ClassVar[str] = "按 id 更新已有项目记忆，并说明修改原因"
    input_type: type[MemoryUpdateToolInput] = MemoryUpdateToolInput
    capabilities: ClassVar[set[ToolCapability]] = {ToolCapability.WRITE}

    async def _impl(
        self, params: MemoryUpdateToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        """直接调用共享 MemoryService 更新，同一条目保留原 ID。"""
        item = await self.service.aupdate(
            params.item_id,
            self.access_policy_factory(context).write_namespace,
            params.patch,
            actor=MemoryActor.AGENT,
            reason=params.reason,
            source_type=MemorySourceType.TOOL_EVENT,
            source_id=context.call_id,
        )
        return await self._write_result(item.namespace, {"item": _item_payload(item)})


class MemoryForgetTool(MemoryTool[MemoryForgetToolInput]):
    """软删除宿主绑定的 namespace 中的条目。"""

    name: ClassVar[str] = "memory_forget"
    description: ClassVar[str] = "按 id 删除不再需要的项目记忆，并说明删除原因"
    input_type: type[MemoryForgetToolInput] = MemoryForgetToolInput
    capabilities: ClassVar[set[ToolCapability]] = {ToolCapability.WRITE}

    async def _impl(
        self, params: MemoryForgetToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        """返回数据库是否实际删除条目，不把未命中伪装成删除成功。"""
        namespace = self.access_policy_factory(context).write_namespace
        deleted = await self.service.aforget(
            params.item_id,
            namespace,
            actor=MemoryActor.AGENT,
            reason=params.reason,
            source_type=MemorySourceType.TOOL_EVENT,
            source_id=context.call_id,
        )
        return await self._write_result(namespace, {"deleted": deleted})


MEMORY_TOOL_CLASSES: dict[str, type[MemoryTool[Any]]] = {
    "memory.search": MemorySearchTool,
    "memory.fetch": MemoryFetchTool,
    "memory.remember": MemoryRememberTool,
    "memory.update": MemoryUpdateTool,
    "memory.forget": MemoryForgetTool,
}


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
    tool_names: Sequence[str] = (),
) -> ToolRegistry:
    """注册选定记忆工具并返回 registry，默认不注册任何工具。

    Args:
        service (MemoryService): 供所有记忆工具共享的服务实例。
        access_policy_factory (MemoryAccessPolicyFactory): 基于工具执行上下文生成
            read/write namespace 分离访问策略的工厂。
        registry (ToolRegistry | None): 要扩展的已有 registry。为 None 时创建新 registry。
        max_result_chars (int): 每个记忆工具允许返回给模型的最大字符数。
        tool_names: 从 MEMORY_TOOL_CLASSES 选择的 builtin 声明名。

    Returns:
        ToolRegistry: 注册完选定记忆工具的 registry。
            如果传入了 `registry`，返回值就是同一个对象，便于和文件工具等其它工具组合注册。
    """
    registry = registry or ToolRegistry()
    for name in tool_names:
        tool_cls = MEMORY_TOOL_CLASSES[name]
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
    return payload


def _hit_payload(hit: MemorySearchHit) -> dict[str, Any]:
    """将进程内搜索结果投影为固定六字段工具输出。"""
    return {
        "item_id": hit.item_id,
        "namespace": hit.namespace,
        "category": hit.category.value,
        "kind": hit.kind.value,
        "snippet": hit.snippet,
        "is_complete": hit.is_complete,
    }
