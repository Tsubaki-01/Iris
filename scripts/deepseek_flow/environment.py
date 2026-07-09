"""DeepSeek live 验证运行环境元数据。"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def collect_run_environment(repo_root: Path) -> dict[str, Any]:
    """收集一次验证运行对应的本地代码与 Python 环境信息。"""
    return {
        "repo_root": str(repo_root),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git": _git_metadata(repo_root),
    }


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    """读取当前 git 版本信息。"""
    commit = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "branch", "--show-current")
    status = _git(repo_root, "status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _git(repo_root: Path, *args: str) -> str:
    """执行只读 git 命令并返回 stdout。"""
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
