"""工具注册表与只读视图。

提供统一的接口管理、检索和过滤工具实例，并在执行上下文中导出对应的 schema 模型。

Example:
    registry = ToolRegistry()
    registry.register(my_tool)
    view = registry.view(allow={"my_tool"})
"""

# region imports
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

# endregion


class ToolRegistry:
    """保存工具实例并导出活动 schema。

    负责集中式状态管理，防止工具名称或别名在运行态冲突。主要通过
    只读的 `ToolRegistryView` 对实际执行层暴露。

    Attributes:
        _tools (dict[str, BaseTool]): 内部存储的具体工具实例，按规范化名称索引。
        _aliases (dict[str, str]): 全局维护的别名到主名称映射表。

    Example:
        registry = ToolRegistry()
        registry.register(tool)
        schemas = registry.active_schemas()
    """

    # ==========================================
    #               Initialization
    # ==========================================
    # region
    def __init__(self) -> None:
        """初始化空注册表。

        准备管理底层状态与别名映射词典。
        """
        self._tools: dict[str, BaseTool] = {}
        self._aliases: dict[str, str] = {}

    # endregion

    # ==========================================
    #               Tool Registration
    # ==========================================
    # region
    def register(self, tool: BaseTool, *, on_conflict: str = "raise") -> None:
        """注册对象式工具。

        确保新工具的主名称与别名与现有系统完全隔离，不产生冲突。

        Args:
            tool (BaseTool): 要注册的工具实例。
            on_conflict (str): 发生同名时的决策策略。

        Raises:
            IrisToolValidationError: 当名称、别名产生冲突或 on_conflict 不为 raise 时。
        """
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
        """将普通函数包装为工具并注册。

        为纯函数封装 CallableTool 以便系统以标准的 BaseTool 生命周期调度调用。

        Args:
            func (Callable[..., Any]): 可以执行核心逻辑的 Python 纯函数。
            name (str | None): 重写的工具主名称。
            description (str | None): 覆盖原函数的 docstring 作为描述。
            input_model (type[BaseModel] | None): 自定义输入验证模型。
            capabilities (set[ToolCapability] | None): 工具特性支持项配置。
            group (str): 工具归属组，有助于切面鉴权与显式过滤。
            deferred (bool): 是否默认隐式，必须明确 allow 才能暴露给模型。
            preset_kwargs (dict[str, Any] | None): 绑定给函数的常数调用参数。

        Returns:
            BaseTool: 包裹原函数并完成全局挂载的新工具。

        Raises:
            IrisToolValidationError: 当工具命名不符合注册表规定产生冲突时。
        """
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

    # endregion

    # ==========================================
    #               Access & Export
    # ==========================================
    # region
    def get(self, name: str) -> BaseTool:
        """通过名称或别名获取工具。

        执行名称解析时，优先检索别名词典确认其重定向的目标。

        Args:
            name (str): 查询用的原名或任意注册别名。

        Returns:
            BaseTool: 命中对应词条的工具对象。

        Raises:
            IrisToolNotFoundError: 当名称不在记录且别名表中也无法追踪时。
        """
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
        """创建只读过滤视图。

        用于生成按会话、按权限过滤后的影子注册表，不改变全局状态。

        Args:
            include_groups (set[str] | None): 必须匹配的组群标签名录，未含其中的组类工具默认剔除。
            allow (set[str] | None): 白名单主名称，强制透传以复活 deferred 或受组过滤清理的记录。
            deny (set[str] | None): 黑名单主名称限制，高优拦截任意试图暴露的对应实例。

        Returns:
            ToolRegistryView: 新配置出炉仅供查询的只读视图实例。
        """
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
        """导出当前活动工具 schema。

        代理通过生成默认全局视图导出兼容具体 Provider 标准的 schema 定义结构体。

        Args:
            provider (str | None): 可选的 LLM 供应商名称（如 openai）。
            api_style (str | None): 具体提供者的特殊调用分格类型定义。

        Returns:
            list[dict[str, object]]: 符合 API 要求的 JSON schema 一维数组。
        """
        return self.view().active_schemas(provider=provider, api_style=api_style)

    # endregion

    # ==========================================
    #               Internal Helpers
    # ==========================================
    # region
    def _active_tools(self) -> list[BaseTool]:
        """返回注册顺序下的全部工具。

        Returns:
            list[BaseTool]: 所有被正确录于当前缓存池的工作工具列表。
        """
        return list(self._tools.values())

    def _validate_available_name(self, name: str) -> None:
        """校验名称未与已注册名称或别名冲突。

        拦截任何会导致当前映射网络受损或死循环的更新。

        Args:
            name (str): 预期录入数据库的工具新分配指称名。

        Raises:
            IrisToolValidationError: 当目标查重在本体或别名任何一面未通过时引爆。
        """
        existing_names = set(self._tools) | set(self._aliases)
        for tool in self._tools.values():
            existing_names.update(tool.definition.aliases)
        if name in existing_names:
            raise IrisToolValidationError("工具名称或别名重复", name=name)

    # endregion


class ToolRegistryView:
    """注册表的只读过滤视图。

    用于动态执行策略并确保不同会话上下文读取自己所需的子集隔离能力。不修改底层表字典结构。

    Attributes:
        registry (ToolRegistry): 提供持久化源库引用的主注册实体。
        include_groups (set[str] | None): 保存待显露组分类信息的合规池。
        allow (set[str]): 工具名称提权的特别免死集合。
        deny (set[str]): 工具名称严厉屏蔽剔除的名单字典。

    Example:
        view = ToolRegistryView(reg, deny={"evil_tool"})
        tools = view.active_tools()
    """

    # ==========================================
    #               Initialization
    # ==========================================
    # region
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        include_groups: set[str] | None = None,
        allow: set[str] | None = None,
        deny: set[str] | None = None,
    ) -> None:
        """初始化过滤视图。

        存储需要进行视图裁剪所需的各项显式参数，为过滤行为固定配置。

        Args:
            registry (ToolRegistry): 已装载内容的母版工具持久层。
            include_groups (set[str] | None): 限定只加载部分合法归组范围。
            allow (set[str] | None): 例外保活白名单列表。
            deny (set[str] | None): 强行拦截不再响应黑名单。
        """
        self.registry = registry
        self.include_groups = include_groups
        self.allow = allow or set()
        self.deny = deny or set()

    # endregion

    # ==========================================
    #               Core Methods
    # ==========================================
    # region
    def get(self, name: str) -> BaseTool:
        """通过原注册表获取工具。

        当前设计直接绕过视图安全配置进行溯源获取底层可用件。

        Args:
            name (str): 全局范围寻址检索识别码信息。

        Returns:
            BaseTool: 无差异透传查询下找到的根记录载体。

        Raises:
            IrisToolNotFoundError: 从主干路由找不到目标条目记录时抛送。
        """
        return self.registry.get(name)

    def active_tools(self) -> list[BaseTool]:
        """按视图规则返回活动工具。

        以流转过滤的形式迭代，先判断黑名单抛弃，再判断组别/延迟性质进行条件排除。

        Returns:
            list[BaseTool]: 顺次通过约束筛选准予通行的合规件清单。
        """
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
        """导出活动工具 schema。

        格式化符合权限标准的工具供外部对接消费。

        Args:
            provider (str | None): 要对齐的指定大模型底层规范字柄。
            api_style (str | None): API 要求的特殊封装架构方式风格。

        Returns:
            list[dict[str, object]]: 生成妥当随时可挂载作为参数的模型集合。
        """
        return [
            _format_schema(tool.definition, provider=provider, api_style=api_style)
            for tool in self.active_tools()
        ]

    # endregion


def _format_schema(
    definition: ToolDefinition,
    *,
    provider: str | None,
    api_style: str | None,
) -> dict[str, object]:
    """按 provider 生成 schema。

    支持对 OpenAI (包含常规和 response), Anthropic 特殊格式的定向翻译操作。

    Args:
        definition (ToolDefinition): 用于提供源元信息的描述定义。
        provider (str | None): 需要对应映射构建配置的外部供应商名称。
        api_style (str | None): 仅给 OpenAI 处理响应定制结构时作为补充区别使用。

    Returns:
        dict[str, object]: 按指派供应商生成的词典数据。若未设定提供商就回滚中继形式。

    Raises:
        IrisToolValidationError: 在风格不明晰或不支持的供应商被硬性匹配时引发。

    Example:
        >>> _format_schema(tool_def, provider="anthropic", api_style=None)
        {'name': '...', 'description': '...', 'input_schema': {...}}
    """
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
