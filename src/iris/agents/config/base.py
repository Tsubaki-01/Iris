"""Agent YAML 配置模型与加载入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ...core import ModelRoute, parse_model_route
from ...exceptions import IrisConfigError, IrisValidationError


class ModelConfig(BaseModel):
    """模型路由配置。

    Attributes:
        provider (str): Provider 名称，例如 `openai`。
        name (str): Provider 下的模型名，例如 `gpt-4o-mini`。
        api_style (str | None): 可选 API 风格。
        base_url (str | None): 可选 provider base URL。
        temperature (float | None): 采样温度。
        top_p (float | None): nucleus sampling 参数。
        max_tokens (int | None): 最大输出 token 数。
        tool_choice (str | dict[str, Any] | None): 工具选择策略。
        response_format (dict[str, Any] | None): 结构化输出配置。
        stream (bool): 是否请求流式响应。
        timeout (float | None): 单次请求超时时间，单位秒。
        provider_options (dict[str, Any]): 少量 provider 专属选项。
        metadata (dict[str, Any]): 请求级元数据。
    """

    provider: str
    name: str
    api_style: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    stream: bool = False
    timeout: float | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("provider", "name")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        """校验模型路由字段非空。"""
        if not value.strip():
            raise ValueError("模型 provider/name 不能为空")
        return value

    def to_model_route(self) -> ModelRoute:
        """转换为现有 `ModelRoute`。

        Returns:
            ModelRoute: Provider 与模型名路由对象。
        """
        return ModelRoute(provider=self.provider, model=self.name)

    def to_llm_request_options(self) -> dict[str, Any]:
        """
        转换为 `LLMRequest` 支持的请求级选项。
        由于 `LLMRequest` 显式要求 `model` 字段，因此不包含 `provider` 和 `name`。
        """
        options = self.model_dump(
            exclude={
                "provider",
                "name",
                "api_style",
                "base_url",
            },
            exclude_none=True,
        )
        provider_options = dict(self.provider_options)
        if self.api_style is not None:
            provider_options["api_style"] = self.api_style
        if provider_options:
            options["provider_options"] = provider_options
        return options


class PythonToolsConfig(BaseModel):
    """Python 引用扩展配置。

    Attributes:
        functions (list[str]): 直接注册的 `module:function` 工具函数引用。
        registrars (list[str]): 接收 `ToolRegistry` 的批量注册函数引用。
    """

    functions: list[str] = Field(default_factory=list)
    registrars: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolsConfig(BaseModel):
    """工具声明配置。

    Attributes:
        builtin (list[str]): 内置工具公开 YAML 名称。
        python (PythonToolsConfig): Python 扩展引用配置。
    """

    builtin: list[str] = Field(default_factory=list)
    python: PythonToolsConfig = Field(default_factory=PythonToolsConfig)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _reject_mixed_python_list(cls, data: Any) -> Any:
        """禁止 `tools.python` 混合列表。"""
        if isinstance(data, dict) and isinstance(data.get("python"), list):
            raise ValueError("tools.python 必须使用 functions/registrars 结构，不能使用混合列表")
        return data


class PermissionsConfig(BaseModel):
    """权限声明配置。

    Attributes:
        workspace (str): Agent 工作区路径。
        writes (Literal["confirm", "allow", "deny"]): 写入策略。
    """

    workspace: str = "."
    writes: Literal["confirm", "allow", "deny"] = "confirm"

    model_config = ConfigDict(extra="forbid", frozen=True)


class SessionConfig(BaseModel):
    """会话持久化配置。

    Attributes:
        backend (Literal["none", "sqlite"]): 会话后端。
        path (str | None): SQLite 文件路径。
    """

    backend: Literal["none", "sqlite"] = "none"
    path: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _default_sqlite_path(self) -> SessionConfig:
        """SQLite 后端未显式配置路径时使用官方默认路径。"""
        if self.backend == "sqlite" and self.path is None:
            return self.model_copy(update={"path": ".iris/session.db"})
        return self


class AgentContextConfig(BaseModel):
    """Agent 引用的 context 配置声明。

    Attributes:
        path (Path): 独立 `context.yaml` 文件路径。
    """

    path: Path

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: Any, info: ValidationInfo) -> Any:
        """校验 context 配置路径非空，并将相对路径解析为绝对路径。"""
        if isinstance(value, str) and not value.strip():
            raise ValueError("context.path 不能为空")
        config_path: Path | None = (info.context or {}).get("config_path")
        if config_path is not None:
            p = Path(value) if isinstance(value, str) else Path(value)
            if not p.is_absolute():
                return (config_path.parent / p).resolve()
        return value


class AgentConfig(BaseModel):
    """Agent YAML 的稳定 Python 配置模型。

    Attributes:
        name (str): Agent 名称。
        model (ModelConfig): 模型配置。
        system (str | None): 简单模式的 system prompt。
        context (AgentContextConfig | None): 结构化 context 配置声明。
        tools (ToolsConfig): 工具配置。
        permissions (PermissionsConfig): 权限配置。
        session (SessionConfig): 会话配置。
    """

    name: str
    model: ModelConfig
    system: str | None = None
    context: AgentContextConfig | None = None
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _validate_prompt_source_keys(cls, data: Any) -> Any:
        """校验 system 和 context 配置键只能二选一。"""
        if not isinstance(data, dict):
            return data
        has_system = "system" in data
        has_context = "context" in data
        if has_system == has_context:
            raise ValueError("system 和 context 必须且只能配置一个")
        return data

    @model_validator(mode="before")
    @classmethod
    def _normalize_model(cls, data: Any) -> Any:
        """支持 `model: provider/name` 短写。"""
        if not isinstance(data, dict):
            return data
        raw_model = data.get("model")
        if isinstance(raw_model, str):
            try:
                route = parse_model_route(raw_model)
            except IrisValidationError as exc:
                raise ValueError(str(exc)) from exc
            data = dict(data)
            data["model"] = {"provider": route.provider, "name": route.model}
        return data

    @field_validator("name", "system")
    @classmethod
    def _validate_text(cls, value: str | None) -> str | None:
        """校验配置文本非空。"""
        if value is not None and not value.strip():
            raise ValueError("字段不能为空")
        return value

    @model_validator(mode="after")
    def _validate_prompt_source(self) -> AgentConfig:
        """校验 simple system 和 structured context 只能二选一。"""
        has_system = self.system is not None
        has_context = self.context is not None
        if has_system == has_context:
            raise ValueError("system 和 context 必须且只能配置一个")
        return self

    def to_model_route(self) -> ModelRoute:
        """返回配置对应的模型路由。

        Returns:
            ModelRoute: Provider 与模型名路由对象。
        """
        return self.model.to_model_route()


def load_agent_config(path: str | Path) -> AgentConfig:
    """从 YAML 文件加载 Agent 配置。

    Args:
        path (str | Path): `agent.yaml` 文件路径。

    Returns:
        AgentConfig: 已校验并规范化的 Agent 配置。

    Raises:
        IrisConfigError: 文件、YAML 或字段校验失败时抛出。
    """
    config_path = Path(path)
    if not config_path.exists():
        raise IrisConfigError("Agent 配置文件不存在", path=str(config_path))
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IrisConfigError("Agent 配置文件读取失败", path=str(config_path)) from exc

    try:
        raw_config = yaml.safe_load(config_text)
    except yaml.YAMLError as exc:
        raise IrisConfigError("Agent 配置 YAML 解析失败", path=str(config_path)) from exc

    if not isinstance(raw_config, dict):
        raise IrisConfigError("Agent 配置顶层必须是对象", path=str(config_path))

    try:
        return AgentConfig.model_validate(raw_config, context={"config_path": config_path})
    except ValidationError as exc:
        raise IrisConfigError("Agent 配置校验失败", path=str(config_path), error=str(exc)) from exc


__all__ = [
    "AgentContextConfig",
    "AgentConfig",
    "ModelConfig",
    "PermissionsConfig",
    "PythonToolsConfig",
    "SessionConfig",
    "ToolsConfig",
    "load_agent_config",
]
