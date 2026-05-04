"""基于 Pydantic Settings 的集中式配置管理。

该模块定义了一个不可变的运行时配置对象，并提供进程级单例生命周期
（`init_config`、`get_config`、`reset`）。配置优先级为：显式关键字参数
> 环境变量 > 字段默认值。

默认不会加载 `.env` 文件（`env_file=None`）。
如有需要，可在调用`init_config` 时传入 `env_file`。

示例:
    import iris

    iris.init_config(api_key="sk-xxx", debug=True)
    # iris.init_config(env_file=".env")  # 可选加载 dotenv
    cfg = iris.get_config()
"""

from __future__ import annotations

# region imports
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import IrisConfigError

# endregion


# ============================================================
# 配置模型
# ============================================================
# region Config
class Config(BaseSettings):
    """从多种来源解析得到的不可变应用配置。

    本类会禁止未知字段，以便在参数拼写错误时快速失败；同时保持实例冻结，
    避免运行时被意外修改。

    Example:
        cfg = Config(api_key="sk-xxx", timeout=15)
    """

    model_config = SettingsConfigDict(
        env_prefix="IRIS_",
        env_file=None,  # 默认不加载 dotenv
        case_sensitive=False,  # 同时接受 IRIS_API_KEY 和 IRIS_api_key。
        extra="forbid",  # 未知关键字参数直接报错。
        frozen=True,  # 将配置视为不可变运行时状态。
    )

    # --- 必填字段 ---
    api_key: str | None = Field(
        default=None,
        description="API 密钥",
        json_schema_extra={"required_runtime": True},
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
        """校验必需配置。"""
        prefix = self.model_config.get("env_prefix", "")
        missing = []

        for field_name, field in type(self).model_fields.items():
            extra = field.json_schema_extra or {}
            if not extra.get("required_runtime", False):
                continue

            value = getattr(self, field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(f"{prefix}{field_name.upper()}")

        if missing:
            raise IrisConfigError(
                "缺少必需配置：\n" + "\n".join(f"  - {m}" for m in missing)
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

    if env_file is None:
        _config = Config(**kwargs)
    else:
        # 构造短生命周期子类注入 env_file，避免依赖私有构造参数。
        base = {k: v for k, v in Config.model_config.items() if k != "env_file"}
        runtime_settings_cls = type(
            "RuntimeConfig",
            (Config,),
            {
                "model_config": SettingsConfigDict(
                    **base,
                    env_file=env_file,
                )
            },
        )

        _config = runtime_settings_cls(**kwargs)
    return _config


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
