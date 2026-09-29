"""离线 provider 驱动真实文件工具与命令示例。"""

import json
import os
import platform
from pathlib import Path

import pytest

from examples.command.docker import run_example as run_docker
from examples.command.native import run_example as run_native
from examples.command.python import run_example as run_python
from iris.agents import load_agent_config

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/command"


def test_command_example_configs_expose_command_and_child_boundaries() -> None:
    """完整配置保留默认命令确认，并区分 child 文件权限与命令权限。"""
    native = load_agent_config(EXAMPLES / "native.yaml")
    docker = load_agent_config(EXAMPLES / "docker.yaml")
    child = load_agent_config(EXAMPLES / "child.yaml")
    assert native.command.mode.value == "native"
    assert native.command.docker is None
    assert native.permissions.execute == docker.permissions.execute == "confirm"
    assert "exec.command" in native.tools.builtin and "exec.command" in docker.tools.builtin
    assert docker.command.docker.network == "none"
    assert docker.command.docker.image == "iris-command:local"
    assert docker.command.docker.cpus == 1
    assert docker.command.docker.memory_mb == 256
    assert docker.command.docker.pids_limit == 64
    assert child.permissions.writes == "deny"
    assert child.permissions.execute == "confirm"
    assert "exec.command" in child.tools.builtin
    assert "command" not in child.model_fields_set


@pytest.mark.asyncio
async def test_native_example_writes_executes_and_reads_in_unicode_workspace(
    tmp_path: Path,
) -> None:
    """使用实际平台 shell，原生文件工具写入的数据经 exec 后再次由文件工具读取。"""
    report = await run_native(tmp_path / "中文 workspace")
    assert report["shell"] == ("cmd.exe" if platform.system() == "Windows" else "/bin/sh")
    assert report["tool_names"] == ["write_file", "write_file", "exec_command", "read_file"]
    assert report["confirmed_tools"] == ["exec_command"]
    assert report["output"] == {"rows": 3, "total": 12}
    assert '"total": 12' in report["file_tool_read"]


@pytest.mark.asyncio
async def test_python_example_repairs_error_and_publishes_report_copy(tmp_path: Path) -> None:
    """真实错误进入后续请求，修正后的报告通过发布工具形成独立副本。"""
    config = load_agent_config(EXAMPLES / "python.yaml")
    assert config.tools.builtin == ["file.write", "exec.python", "file.publish"]
    assert config.permissions.execute == "confirm"
    workspace = tmp_path / "Python 中文 workspace"
    report = await run_python(workspace)
    assert report["tool_names"] == ["write_file", "run_python", "run_python", "publish_artifact"]
    assert report["confirmed_tools"] == ["run_python", "run_python"]
    assert report["output"] == {"rows": 3, "total": 12}
    assert "KeyError" in report["error_feedback"] and "Traceback" in report["error_feedback"]
    assert "<iris-python>" in report["error_feedback"]
    source = workspace / "report.json"
    published = Path(report["artifact"]["path"])
    assert published != source and published.is_relative_to(workspace / ".iris" / "tool-results")
    assert report["artifact"]["mime_type"] == "application/json"
    assert report["artifact"]["size_bytes"] == published.stat().st_size
    source.write_text("changed after publication", encoding="utf-8")
    assert json.loads(published.read_text(encoding="utf-8")) == report["output"]
    source.unlink()
    assert json.loads(published.read_text(encoding="utf-8")) == report["output"]


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("IRIS_RUN_DOCKER_EXAMPLES") != "1",
    reason="真实 Docker 示例需显式 IRIS_RUN_DOCKER_EXAMPLES=1 和预备镜像",
)
async def test_docker_example_reuses_service_and_restarts_after_run_failure(tmp_path: Path) -> None:
    """真实 Runner 跨 session 复用、失败停止、WAITING 保持和重启文件保留。"""
    report = await run_docker(tmp_path / "中文 workspace")
    assert report["session_a"] == report["session_b"] == {"total": 12}
    assert report["waiting_survived"] is True
    assert report["failure_reason"] == "failed"
    assert report["after_stop"] == {"dependency_retained": True, "service_running": False}
    assert report["after_restart"] == {"total": 12}
    assert report["confirmed_tools"] == ["exec_command"] * 5
    assert report["shell"] == "/bin/sh"
