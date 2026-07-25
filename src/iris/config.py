"""基于 Pydantic Settings 的集中式配置管理。

该模块定义了一个不可变的运行时配置对象，并提供进程级单例生命周期
（`init_config`、`get_config`、`reset`）。配置优先级为：显式关键字参数
> 环境变量 > `.env` 文件 > 字段默认值。

默认不会加载 `.env` 文件（`env_file=None`）。
如有需要，可在调用`init_config` 时传入 `env_file`。
Provider API key 使用 `IRIS_PROVIDER_API_KEYS__{PROVIDER}` 这类 nested env。

示例:
    import iris

    iris.init_config(api_key="sk-xxx", debug=True)
    # iris.init_config(env_file=".env")  # 可选加载 dotenv
    cfg = iris.get_config()
"""

from __future__ import annotations

# region imports
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import IrisConfigError

# endregion


# ============================================================
# 配置模型
# ============================================================
# region Config
class ProviderConfig(BaseModel):
    """单个 provider 的非 secret 运行配置。

    Attributes:
        litellm_provider (str): 传给 LiteLLM 的 provider 名称。
        base_url (str | None): OpenAI-compatible 中转站的 endpoint。
        api_style (Literal["chat"]): 当前仅支持 Chat Completion。
        headers (dict[str, str]): 透传给 provider 的额外 headers。
    """

    litellm_provider: str = Field(default="openai", description="LiteLLM provider")
    base_url: str | None = Field(default=None, description="Provider base URL")
    api_style: Literal["chat"] = Field(default="chat", description="API 风格")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Provider 默认 HTTP headers",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class Config(BaseSettings):
    """从多种来源解析得到的不可变应用配置。

    本类会禁止未知字段，以便在参数拼写错误时快速失败；同时保持实例冻结，
    避免运行时被意外修改。

    Example:
        cfg = Config(api_key="sk-xxx", timeout=15)
    """

    model_config = SettingsConfigDict(
        env_prefix="IRIS_",
        env_nested_delimiter="__",
        env_file=None,  # 默认不加载 dotenv
        case_sensitive=False,  # 同时接受 IRIS_API_KEY 和 IRIS_api_key。
        extra="forbid",  # 未知关键字参数直接报错。
        frozen=True,  # 将配置视为不可变运行时状态。
    )

    # --- 必填字段 ---
    api_key: str | None = Field(
        default=None,
        description="API 密钥",
    )
    provider_api_keys: dict[str, str] = Field(
        default_factory=dict,
        description="按 provider 名称归一化后的 API key",
    )
    providers: dict[str, ProviderConfig] = Field(
        default_factory=dict,
        description="按 provider 名称声明的非 secret 运行配置",
    )

    # --- 可选字段 ---
    base_url: str = Field(
        default="https://api.example.com",
        description="API base URL",
    )
    timeout: int = Field(default=30, ge=1, description="Request timeout (s)")
    debug: bool = Field(default=False, description="Debug mode")

    # --- 字段检验 ---
    def model_post_init(self, __context: Any) -> None:
        """归一化配置字段。"""
        api_key = self.api_key.strip() if self.api_key else None
        object.__setattr__(self, "api_key", api_key or None)
        provider_api_keys = {
            provider.lower(): api_key.strip()
            for provider, api_key in self.provider_api_keys.items()
            if api_key and api_key.strip()
        }
        object.__setattr__(self, "provider_api_keys", provider_api_keys)
        object.__setattr__(
            self,
            "providers",
            {provider.lower(): config for provider, config in self.providers.items()},
        )


# endregion

# ============================================================
# 进程级单例
# ============================================================
# region singleton
_config: Config | None = None


def init_config(*, env_file: str | None = None, **kwargs: Any) -> Config:
    """
    全局配置只初始化一次。

    Args:
        env_file (Optional[str]): 可选 dotenv 文件路径。传入后，
            pydantic-settings 会从该文件加载环境变量。
        **kwargs (Any): 字段覆盖值，优先级高于环境变量与默认值。

    Returns:
        Config: 初始化后的不可变配置对象。

    Raises:
        IrisConfigError: 当重复初始化或校验失败时抛出。

    Example::

        import iris
        iris.init_config(api_key="sk-xxx", debug=True)
    """
    global _config
    if _config is not None:
        raise IrisConfigError("配置已初始化；请勿重复调用 init_config()。")

    _validate_explicit_kwargs(kwargs)
    if env_file is None:
        _config = Config(**kwargs)
    else:
        # 构造短生命周期子类注入 env_file，避免依赖私有构造参数。
        base = {k: v for k, v in Config.model_config.items() if k not in {"env_file", "extra"}}
        runtime_settings_cls = type(
            "RuntimeConfig",
            (Config,),
            {
                "model_config": SettingsConfigDict(
                    **base,
                    env_file=env_file,
                    extra="ignore",
                )
            },
        )

        _config = runtime_settings_cls(**kwargs)
    return _config


def _validate_explicit_kwargs(kwargs: dict[str, Any]) -> None:
    """在读取 env_file 前校验显式传入字段名。"""
    unknown_fields = sorted(set(kwargs) - set(Config.model_fields))
    if unknown_fields:
        raise IrisConfigError(
            "未知配置字段：" + ", ".join(unknown_fields),
            fields=unknown_fields,
        )


def get_config() -> Config:
    """返回已初始化的全局配置。

    Returns:
        Config: 当前不可变配置。

    Raises:
        IrisConfigError: 当尚未调用 `init_config` 时抛出。

    Example:
        cfg = get_config()
    """
    if _config is None:
        raise IrisConfigError("配置尚未初始化；请先调用 init_config()。")
    return _config


def reset() -> None:
    """重置全局配置状态，用于测试或重新引导。

    Returns:
        None: 本函数仅修改模块级状态。

    Example:
        reset()
    """
    global _config
    _config = None


def is_config_initialized() -> bool:
    """检查全局配置是否已初始化。

    Returns:
        bool: 调用过 `init_config` 则为 True，否则为 False。

    Example:
        if not is_config_initialized():
            init_config(api_key="sk-xxx")
    """
    return _config is not None


# endregion
