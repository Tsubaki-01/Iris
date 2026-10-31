"""真实 Linux Docker 的成套快照、进程重启与镜像保留验收。"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from examples.command._scripted import ScriptedProvider, done, require_completed, tool
from examples.lifecycle.environment_recovery import _probe_wheel, build_demo_config
from iris.command.models import CommandRequest, CommandScope, PythonCode
from iris.harness import AgentRunner
from iris.harness._recovery_files import SnapshotStorage
from iris.lifecycle import AgentRunRequest, RunPhase

from ..harness.test_environment_restore import vacate
from .test_docker_integration import require_real_docker

__all__ = ["require_real_docker"]


async def _clear_images(endpoint: str, source_id: str) -> None:
    """测试结束后只删除当前临时环境拥有的 tag。"""
    from aiodocker import Docker

    client = Docker(url=endpoint)
    try:
        repository = f"iris-recovery/{source_id}"
        for image in await client.images.list(filters={"reference": [f"{repository}:*"]}):
            for ref in image["RepoTags"] or ():
                if ref.startswith(f"{repository}:"):
                    await client.images.delete(ref, force=False, noprune=True)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_real_snapshot_restart_preserves_dependency_files_and_hitl(tmp_path: Path) -> None:
    from aiodocker import Docker
    from aiodocker.exceptions import DockerError

    workspace, database, directory = (
        tmp_path / "workspace",
        tmp_path / "live.sqlite",
        tmp_path / "snapshots",
    )
    workspace.mkdir()
    wheel = _probe_wheel(workspace)
    config = build_demo_config(workspace, database, "iris-command:local")
    config = config.model_copy(
        update={
            "command": config.command.model_copy(
                update={
                    "docker": config.command.docker.model_copy(
                        update={
                            "cpus": 0.5,
                            "memory_mb": 256,
                            "pids_limit": 64,
                            "environment": {"SNAPSHOT_TEST": "preserved"},
                        }
                    )
                }
            )
        }
    )
    provider = ScriptedProvider()
    runner = AgentRunner.from_config(config, provider=provider)
    service = runner.runtime.environment.command_binding.service
    try:
        provider.steps.extend(
            (
                tool(
                    "install",
                    "run_python",
                    code=(
                        "import subprocess,sys\nfrom pathlib import Path\n"
                        "subprocess.run([sys.executable, '-m', 'pip', 'install', "
                        "'--user', '--no-index', "
                        f"'--no-deps', '--no-compile', '/workspace/{wheel.name}'], check=True)\n"
                        f"Path({wheel.name!r}).unlink()\nPath('snapshot-value.txt').write_text('first')"
                    ),
                ),
                done("installed"),
            )
        )
        first_run = await runner.start(AgentRunRequest(input="install", session_id="files"))
        require_completed(runner, first_run)
        provider.steps.append(tool("ask", "ask_question", question="继续？"))
        waiting = await runner.start(AgentRunRequest(input="ask", session_id="approval"))
        assert waiting.run.phase is RunPhase.WAITING
        first = await runner.save_environment_snapshot(directory)
        assert not wheel.exists()
        provider.steps.extend(
            (
                tool(
                    "change",
                    "run_python",
                    code=(
                        "from pathlib import Path\nimport iris_snapshot_probe\n"
                        "Path(iris_snapshot_probe.__file__).write_text("
                        "\"VALUE = 'second dependency value'\\n\")\n"
                        "Path('snapshot-value.txt').write_text('second')"
                    ),
                ),
                done("changed"),
            )
        )
        second_run = await runner.start(AgentRunRequest(input="change", session_id="files"))
        require_completed(runner, second_run)
        second = await runner.save_environment_snapshot(directory)
        assert not (directory / str(first.snapshot_id)).exists()
        assert second.docker.config.cpus == 0.5 and second.docker.config.memory_mb == 256
        assert second.docker.config.environment["SNAPSHOT_TEST"] == "preserved"
        provider.steps.extend(
            (
                tool(
                    "advance",
                    "run_python",
                    code=(
                        "from pathlib import Path\nimport iris_snapshot_probe\n"
                        "Path(iris_snapshot_probe.__file__).write_text(\"VALUE = 'third'\\n\")\n"
                        "Path('snapshot-value.txt').write_text('third')"
                    ),
                ),
                done("advanced"),
            )
        )
        third_run = await runner.start(AgentRunRequest(input="advance", session_id="files"))
        require_completed(runner, third_run)
        assert await runner.recover(third_run.run.run_id) == third_run
        assert (workspace / "snapshot-value.txt").read_text() == "third"
        host_config = tmp_path / "host.yaml"
        host_config.write_text(yaml.safe_dump(config.model_dump(mode="json", exclude_none=True)))
        await runner.aclose()
        vacate(runner, tmp_path)
        process = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "-m",
                "tests.harness._environment_restart_worker",
                str(host_config),
                str(directory),
                waiting.run.run_id,
                "--probe",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        assert process.returncode == 0, process.stderr
        report = json.loads(process.stdout.splitlines()[-1])
        assert report["source_id"] == second.source_id
        assert report["file"] == "second"
        assert set(report["sessions"]) == {"files", "approval"}
        assert report["interaction_id"] == waiting.pending_interaction.interaction_id
        assert report["steps_before"] == waiting.run.usage.model_steps_committed
        assert report["steps_after"] == report["steps_before"] + 1
        assert report["probe"]["value"] == "second dependency value"
        assert report["probe"]["path"].startswith("/tmp/.local/")
        assert report["probe"]["env"] == "preserved"
        assert report["probe"]["user"] == second.docker.user
        latest = SnapshotStorage(directory).load_latest()
        assert str(latest.snapshot_id) == report["latest"]
        client = Docker(url=second.docker.config.endpoint)
        try:
            await client.images.inspect(report["latest_image"])
            await client.images.inspect("iris-command:local")
            with pytest.raises(DockerError) as missing:
                await client.images.inspect(report["deferred_image"])
            assert missing.value.status == 404
        finally:
            await client.close()
    finally:
        await runner.aclose()
        await _clear_images(service._sandbox._endpoint, runner.store.source_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["absent", "stopped"])
async def test_real_absent_and_stopped_snapshots_keep_original_state(
    tmp_path: Path, state: str
) -> None:
    workspace, database, directory = (
        tmp_path / "workspace",
        tmp_path / "live.sqlite",
        tmp_path / "snapshots",
    )
    config = build_demo_config(workspace, database, "iris-command:local")
    runner = AgentRunner.from_config(config, provider=ScriptedProvider())
    service = runner.runtime.environment.command_binding.service
    restored = None
    try:
        if state == "stopped":
            outcome = await service.execute(
                CommandScope("files", "files"),
                CommandRequest(
                    uuid4().hex,
                    PythonCode(
                        "from pathlib import Path; Path('/tmp/installed.txt').write_text('kept')"
                    ),
                    workspace,
                    10,
                ),
            )
            assert outcome.exit_code == 0, outcome.stderr
            await service.stop(CommandScope("files", "files")).wait_drained()
            before = await service._sandbox.container.show()
        saved = await runner.save_environment_snapshot(directory)
        if state == "absent":
            assert service._sandbox.container is None
        else:
            after = await service._sandbox.container.show()
            assert not after["State"]["Running"] and not after["State"]["Paused"]
            assert before["State"]["StartedAt"] == after["State"]["StartedAt"]
        await runner.aclose()
        vacate(runner, tmp_path)
        restored = await AgentRunner.from_environment_snapshot(
            config, directory, provider=ScriptedProvider()
        )
        assert restored.runtime.environment.command_binding.service._sandbox.container is None
        code = (
            "from pathlib import Path; print(Path('/tmp/installed.txt').read_text())"
            if state == "stopped"
            else "print('base image')"
        )
        outcome = await restored.runtime.environment.command_binding.service.execute(
            CommandScope("probe", "files"),
            CommandRequest(uuid4().hex, PythonCode(code), workspace, 10),
        )
        assert outcome.exit_code == 0, outcome.stderr
        assert outcome.stdout.strip() == ("kept" if state == "stopped" else "base image")
        assert restored.store.source_id == saved.source_id
    finally:
        if restored is not None:
            await restored.aclose()
        await runner.aclose()
        await _clear_images(service._sandbox._endpoint, runner.store.source_id)


@pytest.mark.asyncio
async def test_real_example_cli_saves_then_restores(tmp_path: Path) -> None:
    workspace, database, directory = (
        tmp_path / "workspace",
        tmp_path / "live.sqlite",
        tmp_path / "snapshots",
    )
    arguments = [
        "--workspace",
        str(workspace),
        "--database",
        str(database),
        "--snapshot-directory",
        str(directory),
        "--run-docker",
    ]
    saved = None
    try:
        process = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "examples.lifecycle.environment_recovery", "save", *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
        )
        assert process.returncode == 0, process.stderr
        saved = SnapshotStorage(directory).load_latest()
        assert saved is not None
        assert workspace.is_relative_to(tmp_path) and database.is_relative_to(tmp_path)
        workspace.rename(tmp_path / "original-workspace")
        database.rename(tmp_path / "original.sqlite")
        process = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "-m",
                "examples.lifecycle.environment_recovery",
                "restore",
                *arguments,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
        )
        assert process.returncode == 0, process.stderr
        report = json.loads(process.stdout.splitlines()[-1])
        assert report["stop_reason"] == "completed" and report["file"] == "saved"
    finally:
        if saved is not None:
            await _clear_images(saved.docker.config.endpoint, saved.source_id)
