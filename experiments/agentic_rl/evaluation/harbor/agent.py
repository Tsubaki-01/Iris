"""Harbor 0.24.0 installed agent：宿主调度，容器内独立执行 Iris。"""

from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path, PurePosixPath
from typing import Any, cast

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent, NonZeroAgentExitCodeError
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field

from .results import token_usage

INSTALL = PurePosixPath("/installed-agent/iris")


class IrisOptions(InstalledAgentOptions):
    """由 Harbor 在 CLI/构造边界解析的接入参数。"""

    iris_wheel: Path = Field(description="本仓库构建的 Iris wheel 路径")
    profile_dir: Path | None = Field(default=None, description="自包含的终端 profile 目录")
    max_model_steps: int = Field(default=20, gt=0)
    cleanup_timeout_sec: float = Field(default=30, gt=0)
    iris_revision: str | None = None


class IrisHarborAgent(BaseInstalledAgent):
    """通过本地 wheel 安装 Iris，保持任务解释器与 Iris 解释器独立。"""

    capabilities = AgentCapabilities()
    options_model = IrisOptions

    def __init__(self, logs_dir: Path, **kwargs: Any) -> None:
        """接收明确安装材料；其余模型与环境参数交给 Harbor。"""
        super().__init__(logs_dir=logs_dir, **kwargs)
        if self.model_name is None:
            raise ValueError("Iris requires --model provider/model")
        options = cast(IrisOptions, self.options)
        self.iris_wheel = options.iris_wheel.resolve()
        self.profile_dir = (
            options.profile_dir.resolve()
            if options.profile_dir is not None
            else Path(__file__).parent / "profiles" / "terminal"
        )
        self.max_model_steps = options.max_model_steps
        self.cleanup_timeout_sec = options.cleanup_timeout_sec
        self.iris_revision = options.iris_revision
        self._version = self.iris_wheel.name.split("-")[1]
        self._workspace: str

    @staticmethod
    def name() -> str:
        """返回稳定的 Agent 名称。"""
        return "iris"

    @property
    def iris_logs(self) -> PurePosixPath:
        """本 trial 的环境内日志目录。"""
        return self.environment_logs_dir / "iris"

    async def install(self, environment: BaseEnvironment) -> None:
        """安装独立 Python 与本仓库 wheel，上传原始 profile 后再由容器解析。"""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        pwd = await self.exec_as_agent(environment, "pwd -P")
        self._workspace = pwd.stdout.strip()
        await self.ensure_system_dependencies(environment, ("curl", "ca_certificates"))
        await self.exec_as_root(
            environment,
            f"mkdir -p {INSTALL} {self.iris_logs}",
        )
        await environment.upload_file(self.iris_wheel, str(INSTALL / self.iris_wheel.name))
        await environment.upload_dir(Path(__file__).parent.parent, str(INSTALL / "evaluation"))
        await environment.upload_dir(self.profile_dir, str(INSTALL / "profile"))
        owner = shlex.quote(str(environment.default_user or "root"))
        await self.exec_as_root(
            environment,
            f"chown -R {owner} {INSTALL} {self.iris_logs}",
        )
        install_command = (
            "set -eu; "
            f"curl -LsSf https://astral.sh/uv/install.sh -o {INSTALL}/install-uv.sh && "
            f"UV_INSTALL_DIR={INSTALL}/uv UV_NO_MODIFY_PATH=1 sh {INSTALL}/install-uv.sh && "
            f"{INSTALL}/uv/uv python install 3.12 && "
            f"{INSTALL}/uv/uv venv {INSTALL}/venv --python 3.12 && "
            f"{INSTALL}/uv/uv pip install --python {INSTALL}/venv/bin/python "
            f"{shlex.quote(str(INSTALL / self.iris_wheel.name))}"
        )
        await self.exec_as_agent(
            environment,
            f"({install_command}) > {self.iris_logs}/install.log 2>&1",
            env={
                "UV_PYTHON_INSTALL_DIR": str(INSTALL / "python"),
                "UV_CACHE_DIR": str(INSTALL / "cache"),
            },
        )

    async def _read_json(self, environment: BaseEnvironment, filename: str) -> dict[str, Any]:
        """读取 runner 写入的原子控制/结果文件。"""
        result = await environment.exec(
            command=f"cat {shlex.quote(str(self.iris_logs / filename))}",
            timeout_sec=5,
        )
        if result.return_code != 0:
            raise RuntimeError(f"Iris did not leave {filename}")
        return json.loads(result.stdout or "")

    async def _stop(self, environment: BaseEnvironment) -> None:
        """只停止本 invocation 的明确 PID，确认退出与 SDK 收尾。"""
        pid: int | None = None
        try:
            async with asyncio.timeout(self.cleanup_timeout_sec):
                control = await self._read_json(environment, "control.json")
                pid = int(control["pid"])
                if pid <= 0:
                    raise RuntimeError("Iris control PID must identify one process")
                await environment.exec(command=f"kill -TERM {pid}", timeout_sec=5)
                while True:
                    alive = await environment.exec(command=f"kill -0 {pid}", timeout_sec=5)
                    if alive.return_code != 0:
                        break
                    await asyncio.sleep(0.1)
                record = await self._read_json(environment, "result.json")
                if record["cleanup_completed"] is not True:
                    raise RuntimeError("Iris resource cleanup did not complete")
        except Exception as error:
            if pid is not None and pid > 0:
                await environment.exec(command=f"kill -KILL {pid}", timeout_sec=5)
            raise RuntimeError("Iris stop/cleanup could not be confirmed") from error

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """保留 typed 退出码；取消必须等容器内 Iris 停止后才能交回 Harbor。"""
        instruction_path = self.logs_dir / "instruction.txt"
        instruction_path.write_text(instruction, encoding="utf-8")
        await environment.upload_file(instruction_path, str(INSTALL / "instruction.txt"))
        arguments = [
            str(INSTALL / "venv/bin/python"),
            "-m",
            "evaluation.harbor.runner",
            "--profile",
            str(INSTALL / "profile/agent.yaml"),
            "--model",
            str(self.model_name),
            "--instruction-file",
            str(INSTALL / "instruction.txt"),
            "--workspace",
            self._workspace,
            "--logs-dir",
            str(self.iris_logs),
            "--max-model-steps",
            str(self.max_model_steps),
        ]
        command = (
            "exec "
            + shlex.join(arguments)
            + f" > {self.iris_logs}/stdout.log 2> {self.iris_logs}/stderr.log"
        )
        try:
            result = await environment.exec(
                command=command,
                cwd=self._workspace,
                env={"PYTHONPATH": str(INSTALL)},
            )
            record = await self._read_json(environment, "result.json")
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(self._stop(environment))
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            cleanup.result()
            raise
        if result.return_code == 70 or record["cleanup_completed"] is not True:
            raise RuntimeError("Iris resource cleanup failed")
        if result.return_code == 1:
            raise NonZeroAgentExitCodeError(f"Iris stopped: {record.get('stop_reason')}")
        if result.return_code != 0:
            raise RuntimeError(f"Unexpected Iris process exit: {result.return_code}")
        # 保持 context 为空，Harbor 同步日志后才会触发 post-run 回填。

    def populate_context_post_run(self, context: AgentContext) -> None:
        """从已同步的原生记录填入用量，未知费用和 cache 保持空值。"""
        result_path = self.logs_dir / "iris" / "result.json"
        if not result_path.exists():
            context.metadata = {"iris_revision": self.iris_revision, "iris_result": "missing"}
            return
        record = json.loads(result_path.read_text(encoding="utf-8"))
        context.n_input_tokens, context.n_output_tokens = token_usage(record)
        context.metadata = {"iris_revision": self.iris_revision, "iris": record}
