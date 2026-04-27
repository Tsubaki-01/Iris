"""为 Iris 运行时配置日志系统。

本模块定义通用的控制台与文件日志格式
模块导入时完成初始化。

Exports:
    logger: 控制台与文件双输出的日志器实例

Example:
    from iris.log import logger

    logger.info("This is an info message.")
    logger.error("This is an error message.")
"""

# region imports
from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

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

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} | {message}"
)
# endregion


def setup_logger(log_dir: Union[str, Path, None] = None) -> None:
    """配置 Loguru 日志系统。

    系统配置了两个文件日志输出端，分别设置不同的日志保留周期。

    Args:
        log_dir (Union[str, Path, None]): 日志文件的目标目录。当设置为
        'None'时，默认在当前工作目录下创建log文件夹。

    Example:
        >>> setup_logger("./log")
    """
    log_path = Path(log_dir) if log_dir else Path.cwd() / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level="DEBUG",
        format=_CONSOLE_FORMAT,
        colorize=True,
    )

    logger.add(
        log_path / "runtime.log",
        level="INFO",
        format=_FILE_FORMAT,
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
        enqueue=True,
    )

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


setup_logger()
