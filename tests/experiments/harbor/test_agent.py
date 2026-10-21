"""验证 Harbor 0.24.0 adapter 契约，环境操作使用确定性替身。"""

# ruff: noqa: E402
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")

from evaluation.harbor.agent import IrisHarborAgent
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext


class Environment:
    """提供 adapter 所需的准确异步环境操作，并保留调用参数。"""

    default_user = "agent"

    def __init__(self, *, code: int = 0, cleanup: bool = True) -> None:
        self.code = code
        self.record: dict[str, Any] = {
            "cleanup_completed": cleanup,
            "stop_reason": "completed",
            "trace_status": "complete",
        }
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.uploads: list[tuple[Path, str]] = []
        self.running = asyncio.Event()
        self.block = False
        self.alive = False
        self.control_available = True

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """记录上传，验证源确实存在。"""
        assert Path(source_path).is_file()
        self.uploads.append((Path(source_path), target_path))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """记录源码或原始 profile 上传。"""
        assert Path(source_dir).is_dir()
        self.uploads.append((Path(source_dir), target_dir))

    async def exec(self, command: str, **kwargs: Any) -> ExecResult:
        """模拟进程返回、结果文件和 PID 控制，不解析异常字符串。"""
        self.commands.append((command, kwargs))
        if command.endswith("pwd -P"):
            return ExecResult(return_code=0, stdout="/tasks/project with spaces\n")
        if command.startswith("cat ") and "control.json" in command:
            return ExecResult(
                return_code=0 if self.control_available else 1, stdout=json.dumps({"pid": 123})
            )
        if command.startswith("cat ") and "result.json" in command:
            return ExecResult(return_code=0, stdout=json.dumps(self.record))
        if command.startswith("kill -TERM"):
            self.alive = False
            return ExecResult(return_code=0)
        if command.startswith("kill -0"):
            return ExecResult(return_code=0 if self.alive else 1)
        if "-m evaluation.harbor.runner" in command:
            self.running.set()
            if self.block:
                self.alive = True
                await asyncio.Event().wait()
            return ExecResult(return_code=self.code, stdout="", stderr="")
        return ExecResult(return_code=0)


@pytest.fixture
def agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> IrisHarborAgent:
    """只替换 Harbor 自身的软件包准备，不跳过 adapter 安装流程。"""
    wheel = tmp_path / "iris-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"fixture")
    instance = IrisHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/test-model",
        iris_wheel=str(wheel),
        iris_revision="test-revision",
        extra_env={"IRIS_PROVIDER_API_KEYS__OPENAI": "test-key"},
    )
    monkeypatch.setattr(instance, "ensure_system_dependencies", AsyncMock())
    return instance


@pytest.mark.asyncio
async def test_install_and_run_keep_task_python_and_cwd(agent: IrisHarborAgent) -> None:
    environment = Environment()
    await agent.install(environment)
    context = AgentContext()
    await agent.run("包含引号 ' 和 $ 的任务", environment, context)
    assert context.is_empty()
    assert agent.version() == "0.1.0"
    commands = [command for command, _ in environment.commands]
    assert not any("activate" in command or "export PATH=" in command for command in commands)
    command, options = next(
        item for item in environment.commands if "-m evaluation.harbor.runner" in item[0]
    )
    assert options["cwd"] == "/tasks/project with spaces"
    assert options["env"] == {"PYTHONPATH": "/installed-agent/iris"}
    assert "包含引号" not in command
    assert "/venv/bin/python" in command
    assert any(target.endswith("instruction.txt") for _, target in environment.uploads)
    assert agent.extra_env["IRIS_PROVIDER_API_KEYS__OPENAI"] == "test-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,cleanup,exception",
    [
        (1, True, NonZeroAgentExitCodeError),
        (70, False, RuntimeError),
        (9, True, RuntimeError),
        (0, False, RuntimeError),
    ],
)
async def test_exit_channels(
    agent: IrisHarborAgent, code: int, cleanup: bool, exception: type[Exception]
) -> None:
    environment = Environment(code=code, cleanup=cleanup)
    await agent.install(environment)
    with pytest.raises(exception) as caught:
        await agent.run("输入", environment, AgentContext())
    if exception is RuntimeError:
        assert not isinstance(caught.value, NonZeroAgentExitCodeError)


@pytest.mark.asyncio
async def test_partial_export_does_not_change_success(agent: IrisHarborAgent) -> None:
    environment = Environment()
    environment.record["trace_status"] = "partial"
    await agent.install(environment)
    await agent.run("输入", environment, AgentContext())


@pytest.mark.asyncio
@pytest.mark.parametrize("has_control", [True, False])
async def test_cancellation_confirms_exit_and_cleanup(
    agent: IrisHarborAgent, has_control: bool
) -> None:
    environment = Environment()
    environment.block = True
    environment.control_available = has_control
    await agent.install(environment)
    task = asyncio.create_task(agent.run("输入", environment, AgentContext()))
    await environment.running.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError if has_control else RuntimeError):
        await task
    if has_control:
        assert not environment.alive
        assert any(command.startswith("kill -TERM 123") for command, _ in environment.commands)


def test_post_run_fills_usage_and_metadata(agent: IrisHarborAgent) -> None:
    folder = agent.logs_dir / "iris"
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(
        json.dumps(
            {
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "compaction": {"input_tokens": 7, "output_tokens": 5},
                },
                "stop_reason": "completed",
                "trace_status": "partial",
                "cleanup_completed": True,
            }
        ),
        encoding="utf-8",
    )
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert (context.n_input_tokens, context.n_output_tokens) == (10, 7)
    assert context.n_cache_tokens is None and context.cost_usd is None
    assert context.metadata["iris_revision"] == "test-revision"
    assert context.metadata["iris"]["trace_status"] == "partial"
