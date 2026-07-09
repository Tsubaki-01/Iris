"""DeepSeek live 验证脚本配置与日志辅助。"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from iris.config import get_config, init_config, is_config_initialized
from iris.exceptions import IrisConfigError
from iris.log import logger, setup_logger

from .bootstrap import ROOT
from .constants import LOCAL_ENV_FILES


def init_local_config(base_dir: Path = ROOT) -> bool:
    """通过 Iris 集中配置初始化脚本运行配置。

    Args:
        base_dir: 仓库根目录或测试用目录。

    Returns:
        配置初始化成功则为 True，缺少必需配置则为 False。
    """
    if is_config_initialized():
        return True

    for env_path in _local_env_paths(base_dir):
        if _try_init_config(env_path):
            return True
    return _try_init_config(None)


def resolve_api_key() -> str | None:
    """解析当前配置中 DeepSeek 可用的 API key。"""
    if not is_config_initialized():
        return None
    config = get_config()
    return config.provider_api_keys.get("deepseek") or config.api_key


def setup_flow_logging(log_dir: Path) -> None:
    """配置 DeepSeek 验证脚本的文件日志。

    Args:
        log_dir: 日志输出目录。
    """
    setup_logger(log_dir)
    logger.info("deepseek.logging.configured log_dir={}", log_dir)


def mask_secret(value: str | None) -> str:
    """返回适合打印的 secret 掩码。"""
    if not value:
        return "<missing>"
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:3]}...{value[-4:]}"


def _safe_error_message(error: Exception) -> str:
    """返回不会泄露当前 API key 的异常摘要。"""
    return _redact_current_api_key(str(error))


def _redact_current_api_key(message: str) -> str:
    """从文本中移除当前配置的 API key。"""
    api_key = resolve_api_key()
    if api_key:
        return message.replace(api_key, mask_secret(api_key))
    return message


def _local_env_paths(base_dir: Path) -> list[Path]:
    """返回按优先级存在的本地 env 文件路径。"""
    return [
        base_dir / file_name
        for file_name in LOCAL_ENV_FILES
        if (base_dir / file_name).exists()
    ]


def _try_init_config(env_path: Path | None) -> bool:
    """尝试用指定 env 文件初始化 Iris 配置。"""
    try:
        if env_path is None:
            init_config()
        else:
            init_config(env_file=str(env_path))
    except (IrisConfigError, ValidationError):
        return False
    return True
