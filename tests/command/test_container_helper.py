"""Linux 一次性助手的真实进程测试；不以 Windows mock 冒充进程组证据。"""

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from iris.command import _container_helper

pytestmark = pytest.mark.skipif(os.name == "nt", reason="助手运行于 Linux 容器，需 POSIX 进程组")


def _run(tmp_path: Path, code: str, timeout: float = 3) -> tuple[dict[str, object], bytes]:
    script = tmp_path / "command.py"
    script.write_text(code, encoding="utf-8")
    result_path = tmp_path / "result.json"
    helper = subprocess.run(
        [
            sys.executable,
            _container_helper.__file__,
            "exec " + shlex.join([sys.executable, "-u", str(script)]),
            str(timeout),
            str(result_path),
        ],
        capture_output=True,
        timeout=timeout + 5,
    )
    assert helper.returncode == 0, helper.stderr
    return json.loads(result_path.read_text(encoding="utf-8")), helper.stdout


@pytest.mark.parametrize("code", [0, 7, 124, 137])
def test_real_return_codes_are_distinct_from_helper_timeout(tmp_path: Path, code: int) -> None:
    result, output = _run(tmp_path, f"print('ready'); raise SystemExit({code})")
    assert result == {"reason": "exited", "returncode": code}
    assert output.strip() == b"ready"


def test_timeout_is_preserved_when_command_handles_term_with_zero_exit(tmp_path: Path) -> None:
    result, _ = _run(
        tmp_path,
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "print('ready', flush=True)\ntime.sleep(30)\n",
        timeout=0.5,
    )
    assert result == {"reason": "timed_out", "returncode": 0}


def test_timeout_kills_same_group_descendant_after_shell_exits(tmp_path: Path) -> None:
    child = tmp_path / "descendant.py"
    child.write_text(
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('child ready', flush=True)\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    result, output = _run(
        tmp_path,
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-u', {str(child)!r}])\n"
        "time.sleep(30)\n",
        timeout=0.5,
    )
    assert result["reason"] == "timed_out"
    assert b"child ready" in output
    # communicate 已取得 EOF，证明继承管道且忽略 TERM 的同组程序也已退出。


def test_normal_exit_publishes_result_without_waiting_for_background_pipe(tmp_path: Path) -> None:
    marker = tmp_path / "background.pid"
    result_path = tmp_path / "result.json"
    command = tmp_path / "command.py"
    command.write_text(
        "import subprocess, sys, pathlib\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid))\n",
        encoding="utf-8",
    )
    helper = subprocess.Popen(
        [
            sys.executable,
            _container_helper.__file__,
            shlex.join([sys.executable, str(command)]),
            "5",
            str(result_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert helper.wait(timeout=3) == 0
        assert json.loads(result_path.read_text()) == {"reason": "exited", "returncode": 0}
        os.kill(int(marker.read_text()), 0)
    finally:
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if helper.poll() is None:
            helper.kill()
        helper.communicate(timeout=3)


def test_large_output_does_not_change_control_result(tmp_path: Path) -> None:
    started = time.monotonic()
    result, output = _run(tmp_path, "import sys; sys.stdout.buffer.write(b'x' * (2 * 1024 * 1024))")
    assert len(output) == 2 * 1024 * 1024
    assert result == {"reason": "exited", "returncode": 0}
    assert time.monotonic() - started < 5
