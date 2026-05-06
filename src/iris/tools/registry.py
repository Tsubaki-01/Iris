"""工具注册表与只读视图。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from ..exceptions import IrisToolNotFoundError, IrisToolValidationError
from .base import BaseTool, CallableTool, ToolCapability, ToolDefinition
from .schema import (
    to_anthropic_tool_schema,
    to_openai_chat_tool_schema,
    to_openai_responses_tool_schema,
)


class ToolRegistry:
    """保存工具实例并导出活动 schema。"""

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._tools: dict[str, BaseTool] = {}
        self._aliases: dict[str, str] = {}

    def register(self, tool: BaseTool, *, on_conflict: str = "raise") -> None:
        """注册对象式工具。"""
        if on_conflict != "raise":
            raise IrisToolValidationError("阶段 1 只支持 raise 冲突策略", on_conflict=on_conflict)
        self._validate_available_name(tool.definition.name)
        for alias in tool.definition.aliases:
            self._validate_available_name(alias)
        self._tools[tool.definition.name] = tool
        for alias in tool.definition.aliases:
            self._aliases[alias] = tool.definition.name

    def register_function(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        input_model: type[BaseModel] | None = None,
        capabilities: set[ToolCapability] | None = None,
        group: str = "core",
        deferred: bool = False,
        preset_kwargs: dict[str, Any] | None = None,
    ) -> BaseTool:
        """将普通函数包装为工具并注册。"""
        tool = CallableTool(
            func,
            name=name,
            description=description,
            input_model=input_model,
            preset_kwargs=preset_kwargs,
            capabilities=capabilities,
            group=group,
            deferred=deferred,
        )
        self.register(tool)
        return tool

    def get(self, name: str) -> BaseTool:
        """通过名称或别名获取工具。"""
        canonical_name = self._aliases.get(name, name)
        try:
            return self._tools[canonical_name]
        except KeyError as exc:
            raise IrisToolNotFoundError(f"工具不存在: {name}", tool_name=name) from exc

    def view(
        self,
        *,
        include_groups: set[str] | None = None,
        allow: set[str] | None = None,
        deny: set[str] | None = None,
    ) -> ToolRegistryView:
        """创建只读过滤视图。"""
        return ToolRegistryView(
            self,
            include_groups=include_groups,
            allow=allow,
            deny=deny,
        )

    def active_schemas(
        self,
        *,
        provider: str | None = None,
        api_style: str | None = None,
    ) -> list[dict[str, object]]:
        """导出当前活动工具 schema。"""
        return self.view().active_schemas(provider=provider, api_style=api_style)

    def _active_tools(self) -> list[BaseTool]:
        """返回注册顺序下的全部工具。"""
        return list(self._tools.values())

    def _validate_available_name(self, name: str) -> None:
        """校验名称未与已注册名称或别名冲突。"""
        existing_names = set(self._tools) | set(self._aliases)
        for tool in self._tools.values():
            existing_names.update(tool.definition.aliases)
        if name in existing_names:
            raise IrisToolValidationError("工具名称或别名重复", name=name)


class ToolRegistryView:
    """注册表的只读过滤视图。"""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        include_groups: set[str] | None = None,
        allow: set[str] | None = None,
        deny: set[str] | None = None,
    ) -> None:
        """初始化过滤视图。"""
        self.registry = registry
        self.include_groups = include_groups
        self.allow = allow or set()
        self.deny = deny or set()

    def get(self, name: str) -> BaseTool:
        """通过原注册表获取工具。"""
        return self.registry.get(name)

    def active_tools(self) -> list[BaseTool]:
        """按视图规则返回活动工具。"""
        tools: list[BaseTool] = []
        for tool in self.registry._active_tools():
            name = tool.definition.name
            if name in self.deny:
                continue
            if self.include_groups is not None and tool.definition.group not in self.include_groups:
                if name not in self.allow:
                    continue
            if tool.definition.deferred and name not in self.allow:
                continue
            tools.append(tool)
        return tools

    def active_schemas(
        self,
        *,
        provider: str | None = None,
        api_style: str | None = None,
    ) -> list[dict[str, object]]:
        """导出活动工具 schema。"""
        return [
            _format_schema(tool.definition, provider=provider, api_style=api_style)
            for tool in self.active_tools()
        ]


def _format_schema(
    definition: ToolDefinition,
    *,
    provider: str | None,
    api_style: str | None,
) -> dict[str, object]:
    """按 provider 生成 schema。"""
    if provider is None:
        return {
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.input_schema,
        }
    provider_name = provider.lower()
    if provider_name == "openai":
        style = api_style or "chat"
        if style == "chat":
            return to_openai_chat_tool_schema(definition)
        if style == "responses":
            return to_openai_responses_tool_schema(definition)
        raise IrisToolValidationError("不支持的 OpenAI API 风格", api_style=style)
    if provider_name == "anthropic":
        return to_anthropic_tool_schema(definition)
    raise IrisToolValidationError("不支持的工具 schema provider", provider=provider)
