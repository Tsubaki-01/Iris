"""延迟工具发现与本地搜索工具。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from ..exceptions import IrisToolValidationError
from ..message import TextBlock
from .base import BaseTool, ToolDefinition, ToolExecutionContext, ToolResult


class ToolSearchInput(BaseModel):
    """tool_search 的输入参数。"""

    query: str
    include_groups: list[str] | None = None
    limit: int = 10

    def normalized_groups(self) -> set[str] | None:
        """返回去重后的组过滤集合。"""
        if self.include_groups is None:
            return None
        return set(self.include_groups)


class DeferredToolIndex:
    """为 deferred 工具提供轻量本地搜索索引。"""

    def __init__(self) -> None:
        """初始化空索引。"""
        self._definitions: list[ToolDefinition] = []

    def build(self, tools: Iterable[BaseTool]) -> None:
        """用工具集合重建索引。

        Args:
            tools (Iterable[BaseTool]): 注册表中的工具对象。
        """
        self._definitions = [tool.definition for tool in tools if tool.definition.deferred]

    def search(
        self,
        query: str,
        *,
        include_groups: set[str] | None = None,
        limit: int = 10,
    ) -> list[ToolDefinition]:
        """按名称、描述、组和标签搜索 deferred 工具。

        Args:
            query (str): 非空搜索词。
            include_groups (set[str] | None): 可选组过滤。
            limit (int): 最大返回数量。

        Returns:
            list[ToolDefinition]: 相关性最高的候选定义。

        Raises:
            IrisToolValidationError: 查询为空或 limit 非法时抛出。
        """
        normalized_query = query.strip().lower()
        if not normalized_query:
            raise IrisToolValidationError("tool_search query 不能为空")
        capped_limit = min(max(limit, 1), 50)
        scored: list[tuple[int, ToolDefinition]] = []
        terms = normalized_query.split()
        for definition in self._definitions:
            if include_groups is not None and definition.group not in include_groups:
                continue
            haystack = _search_text(definition)
            score = sum(1 for term in terms if term in haystack)
            if normalized_query in haystack:
                score += 2
            if score > 0:
                scored.append((score, definition))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [definition for _, definition in scored[:capped_limit]]


class ToolSearchTool(BaseTool):
    """搜索注册表中默认隐藏的 deferred 工具。"""

    def __init__(self, registry: Any) -> None:
        """初始化 tool_search。

        Args:
            registry (Any): 提供 search_deferred 方法的工具注册表。
        """
        self.registry = registry
        self.definition = ToolDefinition(
            name="tool_search",
            description="搜索可按需激活的延迟工具。",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词。"},
                    "include_groups": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选工具组过滤。",
                    },
                    "limit": {"type": "integer", "default": 10, "description": "最大返回数量。"},
                },
                "required": ["query"],
            },
            group="core",
            metadata={
                "examples": [{"input": {"query": "file search"}}],
                "tags": ["discovery"],
                "version": "1.0",
                "deprecated": False,
                "deprecation_message": None,
            },
        )

    @property
    def input_model(self) -> type[BaseModel] | None:
        """返回搜索输入模型。"""
        return ToolSearchInput

    def validate_input(self, params: dict[str, Any]) -> BaseModel | dict[str, Any]:
        """校验 tool_search 输入。"""
        try:
            value = ToolSearchInput.model_validate(params)
        except ValueError as exc:
            raise IrisToolValidationError("tool_search 参数校验失败", error=str(exc)) from exc
        if not value.query.strip():
            raise IrisToolValidationError("tool_search query 不能为空")
        if value.limit < 1:
            raise IrisToolValidationError("tool_search limit 必须大于 0")
        return value

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行 deferred 工具搜索。"""
        del context
        if isinstance(params, ToolSearchInput):
            search_input = params
        else:
            search_input = ToolSearchInput.model_validate(params)
        matches = self.registry.search_deferred(
            search_input.query,
            include_groups=search_input.normalized_groups(),
            limit=search_input.limit,
        )
        tools = [_definition_summary(definition) for definition in matches]
        text = json.dumps({"tools": tools}, ensure_ascii=False, separators=(",", ":"))
        return ToolResult(
            tool_use_id="",
            tool_name=self.name,
            content=[TextBlock(text=text)],
            data={"tools": tools},
        )


def _definition_summary(definition: ToolDefinition) -> dict[str, Any]:
    """生成适合返回给模型的工具摘要。"""
    metadata = definition.metadata
    return {
        "name": definition.name,
        "description": definition.description,
        "group": definition.group,
        "tags": list(metadata.get("tags", [])),
        "deprecated": bool(metadata.get("deprecated", False)),
    }


def _search_text(definition: ToolDefinition) -> str:
    """生成工具搜索文本。"""
    metadata = definition.metadata
    tags = " ".join(str(tag) for tag in metadata.get("tags", []))
    return f"{definition.name} {definition.description} {definition.group} {tags}".lower()
