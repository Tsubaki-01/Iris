"""独立 Python 进程读取整套恢复点并继续 HITL。"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from iris.agents import ToolsConfig
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest
from iris.message import ToolUseBlock

from ..command.test_docker import FakeClient, driver
from .fakes import StaticProvider, text_response, tool_response
from .test_environment_restore import vacate
from .test_recovery_point import snapshot_runner

__all__ = ["driver"]


@pytest.mark.asyncio
async def test_new_process_restores_waiting_state(tmp_path: Path, driver: FakeClient) -> None:
    base = snapshot_runner(tmp_path)
    config = base.runtime.environment.agent_config.model_copy(
        update={"tools": ToolsConfig(builtin=["human.ask"])}
    )
    await base.aclose()
    runner = AgentRunner.from_config(
        config,
        provider=StaticProvider(
            text_response("files"),
            tool_response(
                ToolUseBlock(id="ask", name="ask_question", input={"question": "继续？"})
            ),
        ),
    )
    (tmp_path / "workspace/snapshot-value.txt").write_text("saved")
    await runner.start(AgentRunRequest(input="files", session_id="files"))
    waiting = await runner.start(AgentRunRequest(input="ask", session_id="approval"))
    directory = tmp_path / "snapshots"
    saved = await runner.save_environment_snapshot(directory)
    config_path = tmp_path / "host.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json", exclude_none=True)))
    await runner.aclose()
    vacate(runner, tmp_path)
    process = await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            "-m",
            "tests.harness._environment_restart_worker",
            str(config_path),
            str(directory),
            waiting.run.run_id,
            "--fake-docker",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout.splitlines()[-1])
    assert report["source_id"] == saved.source_id
    assert report["interaction_id"] == waiting.pending_interaction.interaction_id
    assert report["steps_before"] == waiting.run.usage.model_steps_committed
    assert report["steps_after"] == report["steps_before"] + 1
    assert report["file"] == "saved"
    assert set(report["sessions"]) == {"files", "approval"}
