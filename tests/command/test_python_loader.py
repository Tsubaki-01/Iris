"""同源 Python 启动器的真实子进程语义。"""

import subprocess
import sys
from pathlib import Path

import pytest

from iris.command._python import PYTHON_LOADER_SOURCE


def _run(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    source = tmp_path / "source.txt"
    source.write_text(code, encoding="utf-8")
    cwd = tmp_path / "project"
    cwd.mkdir(exist_ok=True)
    (cwd / "project_module.py").write_text("VALUE = 42", encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-u", "-c", PYTHON_LOADER_SOURCE, str(source)],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )


def test_loader_registers_main_future_annotations_and_cwd_import(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "from __future__ import annotations\n"
        "from dataclasses import dataclass, fields\n"
        "from typing import ClassVar\n"
        "import project_module, sys\n"
        "@dataclass\n"
        "class Item:\n"
        "    category: ClassVar[str] = 'ok'\n"
        "    value: int = project_module.VALUE\n"
        "assert sys.modules[__name__].Item is Item\n"
        "assert len(fields(Item)) == 1\n"
        "assert sys.argv == ['<iris-python>']\n"
        "print('中文', Item().value)\n",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "中文 42\n"


def test_loader_does_not_inherit_future_flags(tmp_path: Path) -> None:
    result = _run(
        tmp_path, "def f(value: int) -> str: pass\nassert f.__annotations__['value'] is int"
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("code", ["answer = 1 / 0\n", "def broken(:\n"])
def test_loader_traceback_has_original_line(tmp_path: Path, code: str) -> None:
    result = _run(tmp_path, code)
    assert result.returncode == 1
    assert "<iris-python>" in result.stderr
    assert code.strip() in result.stderr


def test_loader_does_not_reuse_globals_between_processes(tmp_path: Path) -> None:
    assert _run(tmp_path, "remembered = 42").returncode == 0
    result = _run(tmp_path, "print(remembered)")
    assert result.returncode == 1 and "NameError" in result.stderr
