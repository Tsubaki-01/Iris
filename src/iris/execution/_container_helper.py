"""在 Linux 容器内直接执行的标准库助手，不依赖 Iris 的安装。

argv 依次为命令文本、前台期限和框架生成的结果路径。
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_TERM_GRACE_SECONDS = 0.5


def run(command: str, timeout_seconds: float, result_path: Path) -> None:
    """转发输出，等待前台退出，单独记录真实退出与业务期限。

    Args:
        command (str): 交给 /bin/sh 的命令文本。
        timeout_seconds (float): 已由工具边界确定的前台期限。
        result_path (Path): 当前调用独有的临时结果文件。
    """
    process = subprocess.Popen(
        ["/bin/sh", "-c", command],
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    reason = "exited"
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        reason = "timed_out"
        signalled_at = time.monotonic()
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=_TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        # 前台 shell 已退出时，本次组内仍可能有忽略 TERM 的后代。
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            pass
        else:
            remaining = _TERM_GRACE_SECONDS - (time.monotonic() - signalled_at)
            if remaining > 0:
                time.sleep(remaining)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        returncode = process.wait()
    result_path.write_text(
        json.dumps({"reason": reason, "returncode": returncode}), encoding="utf-8"
    )


def main() -> None:
    """消费框架生成的固定 argv；异常由宿主按缺失结果处理。"""
    command, timeout, result_path = sys.argv[1:]
    run(command, float(timeout), Path(result_path))


if __name__ == "__main__":
    main()
