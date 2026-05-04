"""为 Iris 运行时配置日志系统。

本模块定义通用的控制台与文件日志格式。导入模块只暴露 `logger`
与 `setup_logger`，不会自动创建日志目录或注册文件 sink。

Exports:
    logger: Loguru 日志器实例。
    setup_logger: 显式配置 Iris 日志 sink 的函数。

Example:
    from iris.log import logger, setup_logger

    setup_logger("./iris_log")
    logger.info("This is an info message.")
    logger.error("This is an error message.")
"""

# region imports
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# endregion

# ==========================================
#                 Constants
# ==========================================
# region constants
_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<yellow>[{name}]</yellow> | "
    "<level>{message}</level>"
)

_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"

_SINK_IDS: list[int] = []
# endregion


def setup_logger(log_dir: str | Path | None = None) -> None:
    """配置 Loguru 日志系统。

    默认只配置控制台输出。传入 `log_dir` 时，额外配置两个文件日志输出端，
    分别设置不同的日志保留周期。

    Args:
        log_dir (str | Path | None): 日志文件的目标目录。为 `None` 时不创建
            文件日志目录。

    Example:
        >>> setup_logger("./iris_log")
    """
    _remove_managed_sinks()

    _SINK_IDS.append(
        logger.add(
            sys.stderr,
            level="DEBUG",
            format=_CONSOLE_FORMAT,
            colorize=True,
        )
    )

    if log_dir is None:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    _SINK_IDS.append(
        logger.add(
            log_path / "runtime.log",
            level="INFO",
            format=_FILE_FORMAT,
            rotation="00:00",
            retention="30 days",
            encoding="utf-8",
            enqueue=True,
        )
    )

    _SINK_IDS.append(
        logger.add(
            log_path / "error.log",
            level="ERROR",
            format=_FILE_FORMAT,
            rotation="00:00",
            retention="90 days",
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
    )


def _remove_managed_sinks() -> None:
    """移除本模块通过 `setup_logger` 添加的 sink。"""
    while _SINK_IDS:
        sink_id = _SINK_IDS.pop()
        try:
            logger.remove(sink_id)
        except ValueError:
            continue
