"""在 Linux 容器内直接执行的标准库助手，不依赖 Iris 的安装。

argv 依次为同源启动器、载荷种类/内容、前台期限、stdin 路径和结果路径。
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_TERM_GRACE_SECONDS = 0.5


def run(
    loader: str,
    kind: str,
    payload: str,
    timeout_seconds: float,
    stdin_path: Path | None,
    result_path: Path,
) -> None:
    """转发输出，等待前台退出，单独记录真实退出与业务期限。

    Args:
        loader (str): 框架维护的同源 Python 启动器。
        kind (str): shell 或 python。
        payload (str): shell 文本或本次临时源码路径。
        timeout_seconds (float): 已由工具边界确定的前台期限。
        stdin_path (Path | None): 本次二进制输入文件；None 使用 DEVNULL。
        result_path (Path): 当前调用独有的临时结果文件。
    """
    stdin = stdin_path.open("rb") if stdin_path is not None else None
    try:
        process = subprocess.Popen(
            (
                [sys.executable, "-X", "utf8", "-u", "-c", loader, payload]
                if kind == "python"
                else ["/bin/sh", "-c", payload]
            ),
            stdin=stdin if stdin is not None else subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        if stdin is not None:
            stdin.close()
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
    loader, kind, payload, timeout, stdin_path, result_path = sys.argv[1:]
    try:
        run(
            loader,
            kind,
            payload,
            float(timeout),
            Path(stdin_path) if stdin_path else None,
            Path(result_path),
        )
    finally:
        for path in (payload if kind == "python" else "", stdin_path):
            if not path:
                continue
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).debug(
                    "容器临时输入文件删除失败：%s", path, exc_info=True
                )


if __name__ == "__main__":
    main()
