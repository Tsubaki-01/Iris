"""验证 Harbor 0.24.0 adapter 契约，环境操作使用确定性替身。"""

# ruff: noqa: E402
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor")

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from pydantic import ValidationError

from evals.harbor.agent import IrisHarborAgent, IrisOptions


class Environment:
    """提供 adapter 所需的准确异步环境操作，并保留调用参数。"""

    default_user = "agent"

    def __init__(self, *, code: int = 0, cleanup: bool = True, install_code: int = 0) -> None:
        self.code = code
        self.install_code = install_code
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
        if " pip install " in command:
            return ExecResult(return_code=self.install_code, stderr="download failed")
        if "-m evals.harbor.runner" in command:
            self.running.set()
            if self.block:
                self.alive = True
                await asyncio.Event().wait()
            return ExecResult(return_code=self.code, stdout="", stderr="")
        return ExecResult(return_code=0)


@pytest.fixture
def agent(tmp_path: Path) -> IrisHarborAgent:
    """构造完整安装材料，由环境替身记录实际安装命令。"""
    wheel = tmp_path / "iris-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"fixture")
    uv_executable = tmp_path / "uv"
    uv_executable.write_bytes(b"uv fixture")
    requirements_lock = tmp_path / "requirements.lock"
    requirements_lock.write_text("example-dependency==1.0\n", encoding="utf-8")
    instance = IrisHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/test-model",
        iris_wheel=str(wheel),
        uv_executable=str(uv_executable),
        python_version="3.12.0",
        requirements_lock=str(requirements_lock),
        iris_revision="test-revision",
        extra_env={"IRIS_PROVIDER_API_KEYS__OPENAI": "test-key"},
    )
    return instance


@pytest.mark.parametrize("version", ["3.12", "3.12.*", "latest"])
def test_install_requires_exact_python_patch(version: str) -> None:
    """安装契约拒绝滚动 minor 版本和别名。"""
    with pytest.raises(ValidationError) as caught:
        IrisOptions(
            iris_wheel=Path("iris.whl"),
            uv_executable=Path("uv"),
            python_version=version,
            requirements_lock=Path("requirements.lock"),
        )
    assert caught.value.errors()[0]["loc"] == ("python_version",)
    assert caught.value.errors()[0]["type"] == "string_pattern_mismatch"


@pytest.mark.asyncio
async def test_install_uses_local_uv_and_locked_dependencies(agent: IrisHarborAgent) -> None:
    """本地安装材料决定工具和依赖版本，HTTP 重试只交给 uv。"""
    environment = Environment()
    await agent.install(environment)
    targets = {target for _, target in environment.uploads}
    assert "/installed-agent/iris/uv/uv" in targets
    assert "/installed-agent/iris/requirements.lock" in targets
    commands = [command for command, _ in environment.commands]
    assert not any("curl " in command or "apt-get" in command for command in commands)
    install_commands = [
        (command, options)
        for command, options in environment.commands
        if " pip install " in command
    ]
    assert len(install_commands) == 1
    command, options = install_commands[0]
    assert "python install --no-bin 3.12.0" in command
    assert "--python 3.12.0" in command
    assert "--managed-python" in command
    assert "--constraint /installed-agent/iris/requirements.lock" in command
    assert options["env"]["UV_HTTP_RETRIES"] == "2"
    assert options["env"]["UV_PYTHON_INSTALL_DIR"] == "/installed-agent/iris/python"


@pytest.mark.asyncio
async def test_install_failure_does_not_retry_install_or_start_agent(
    agent: IrisHarborAgent,
) -> None:
    """uv 用尽自身下载重试后，接入层直接保留失败，不叠加安装或模型重跑。"""
    environment = Environment(install_code=1)
    with pytest.raises(NonZeroAgentExitCodeError):
        await agent.install(environment)
    commands = [command for command, _ in environment.commands]
    assert sum(" pip install " in command for command in commands) == 1
    assert not any("-m evals.harbor.runner" in command for command in commands)


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
        item for item in environment.commands if "-m evals.harbor.runner" in item[0]
    )
    assert options["cwd"] == "/tasks/project with spaces"
    assert options["env"] == {"PYTHONPATH": "/installed-agent/iris"}
    assert "包含引号" not in command
    assert "/venv/bin/python" in command
    assert any(target.endswith("instruction.txt") for _, target in environment.uploads)
    assert (Path("evals/__init__.py").resolve(), "/installed-agent/iris/evals/__init__.py") in (
        environment.uploads
    )
    assert (Path("evals/harbor").resolve(), "/installed-agent/iris/evals/harbor") in (
        environment.uploads
    )
    assert all(source.resolve() != Path("evals").resolve() for source, _ in environment.uploads)
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
async def test_cancel_during_result_read_still_checks_cleanup(
    agent: IrisHarborAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """进程已退出但收尾回执尚未读完时，超时不能绕过清理失败通道。"""
    environment = Environment(code=70, cleanup=False)
    await agent.install(environment)
    reading = asyncio.Event()
    original = agent._read_json

    async def delayed_read(environment: Environment, filename: str) -> dict[str, Any]:
        if filename == "result.json" and not reading.is_set():
            reading.set()
            await asyncio.Event().wait()
        return await original(environment, filename)

    monkeypatch.setattr(agent, "_read_json", delayed_read)
    task = asyncio.create_task(agent.run("输入", environment, AgentContext()))
    await asyncio.wait_for(reading.wait(), timeout=3)
    task.cancel()
    with pytest.raises(RuntimeError, match="cleanup"):
        await task


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
