"""提供 lifecycle 示例使用的同步等待工具。

该工具保留同步 claimed-tool 路径，供 cancel/recover 示例观察运行中的工具效果。

Example:
    >>> from examples.lifecycle.tools import wait_for_seconds
    >>> wait_for_seconds(1)
    'waited 1 seconds'
"""

# region imports
import sys
import time

from iris.exceptions import IrisToolValidationError

# endregion


def wait_for_seconds(seconds: int) -> str:
    """同步等待指定秒数，并向标准错误输出 lifecycle 起始标记。

    Args:
        seconds (int): 等待秒数，必须是 1 到 60 之间的非 bool 整数。

    Returns:
        str: 完成等待后的英文结果，例如 ``"waited 2 seconds"``。

    Raises:
        IrisToolValidationError: 当 ``seconds`` 不是 1 到 60 之间的非 bool 整数时。

    Notes:
        标准错误会写入并立即刷新
        ``IRIS_EXAMPLE_TOOL_STARTED seconds={seconds}``，供 lifecycle 示例观察工具启动。
    """
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= 60:
        raise IrisToolValidationError("seconds 必须是 1 到 60 的非布尔整数")
    print(
        f"IRIS_EXAMPLE_TOOL_STARTED seconds={seconds}",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(seconds)
    return f"waited {seconds} seconds"


__all__ = ["wait_for_seconds"]
