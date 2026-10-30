"""显式环境恢复与当前逻辑恢复独立。"""

from pathlib import Path

import pytest

from iris.agents import EnvironmentRecoveryConfig
from iris.exceptions import IrisConfigError, IrisRunRecoveryError
from iris.harness import AgentRunner
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import AgentRunRequest, RunPhase
from iris.message import ToolUseBlock
from iris.store import SQLiteStore

from ..command.test_docker import FakeClient, driver
from .fakes import StaticProvider, text_response, tool_response
from .test_recovery_point import snapshot_runner

__all__ = ["driver"]


def vacate(runner: AgentRunner, root: Path) -> None:
    """fixture 将自身目标移到同一临时根目录，保留推进后的工作副本。"""
    workspace = runner.runtime.environment.workspace_root
    assert workspace.is_relative_to(root)
    assert runner.store.path.is_relative_to(root)
    workspace.rename(root / "advanced-workspace")
    runner.store.path.rename(root / "advanced.sqlite")


@pytest.mark.asyncio
async def test_restores_saved_point_while_recover_uses_current_store(
    tmp_path: Path,
    driver: FakeClient,
) -> None:
    runner = snapshot_runner(tmp_path)
    config = runner.runtime.environment.agent_config
    file = tmp_path / "workspace/value.txt"
    file.write_text("saved")
    first = await runner.start(AgentRunRequest(input="first", session_id="s", run_id="first"))
    directory = tmp_path / "snapshots"
    saved = await runner.save_environment_snapshot(directory)
    file.write_text("advanced")
    # 当前逻辑恢复不读取环境包。
    pointer = directory / "latest.json"
    content = pointer.read_text()
    pointer.write_text("invalid")
    assert await runner.recover(first.run.run_id) == first
    pointer.write_text(content)
    await runner.aclose()
    vacate(runner, tmp_path)
    current = StaticProvider(text_response("next"))
    driver.closed = False
    restored = await AgentRunner.from_environment_snapshot(config, directory, provider=current)
    assert not current.requests
    assert file.read_text() == "saved"
    assert restored.store.source_id == saved.source_id
    assert restored.get_result(first.run.run_id) == first
    assert restored.describe_configuration().workspace_root == saved.workspace_root
    await restored.start(AgentRunRequest(input="next", session_id="s", run_id="next"))
    frozen = SQLiteStore(directory / str(saved.snapshot_id) / "lifecycle.sqlite")
    assert frozen.load_run("next") is None
    await restored.aclose()


@pytest.mark.asyncio
async def test_restored_waiting_keeps_interaction_and_budget(
    tmp_path: Path, driver: FakeClient
) -> None:
    base = snapshot_runner(tmp_path)
    from iris.agents import ToolsConfig

    config = base.runtime.environment.agent_config.model_copy(
        update={
            "tools": ToolsConfig(builtin=["human.ask"]),
        }
    )
    await base.aclose()
    driver.closed = False
    runner = AgentRunner.from_config(
        config,
        provider=StaticProvider(
            tool_response(ToolUseBlock(id="ask", name="ask_question", input={"question": "继续？"}))
        ),
    )
    waiting = await runner.start(AgentRunRequest(input="ask", session_id="s"))
    directory = tmp_path / "snapshots"
    await runner.save_environment_snapshot(directory)
    await runner.aclose()
    vacate(runner, tmp_path)
    current = StaticProvider(text_response("done"))
    driver.closed = False
    restored = await AgentRunner.from_environment_snapshot(config, directory, provider=current)
    result = restored.get_result(waiting.run.run_id)
    assert result.pending_interaction == waiting.pending_interaction
    assert result.run.usage == waiting.run.usage
    continued = await restored.resume(
        waiting.run.run_id,
        interaction_id=waiting.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="yes"),
    )
    assert continued.run.phase is RunPhase.TERMINAL
    assert continued.run.usage.model_steps_committed == waiting.run.usage.model_steps_committed + 1
    await restored.aclose()


@pytest.mark.asyncio
async def test_disabled_restore_does_not_read_snapshot(tmp_path: Path, driver: FakeClient) -> None:
    runner = snapshot_runner(tmp_path)
    config = runner.runtime.environment.agent_config.model_copy(
        update={"environment_recovery": EnvironmentRecoveryConfig(enabled=False)}
    )
    with pytest.raises(IrisConfigError):
        await AgentRunner.from_environment_snapshot(
            config, tmp_path / "missing", provider=StaticProvider()
        )
    assert not driver.info_calls
    await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["nonempty", "missing-archive", "missing-image"])
async def test_restore_preconditions_precede_target_writes(
    tmp_path: Path,
    driver: FakeClient,
    problem: str,
) -> None:
    runner = snapshot_runner(tmp_path)
    directory = tmp_path / "snapshots"
    saved = await runner.save_environment_snapshot(directory)
    config = runner.runtime.environment.agent_config
    await runner.aclose()
    if problem != "nonempty":
        vacate(runner, tmp_path)
    if problem == "missing-archive":
        (directory / str(saved.snapshot_id) / "workspace.tar").unlink()
    elif problem == "missing-image":
        driver.image_missing = True
    driver.closed = False
    current = StaticProvider()
    with pytest.raises(IrisRunRecoveryError):
        await AgentRunner.from_environment_snapshot(config, directory, provider=current)
    assert not current.requests
    if problem != "nonempty":
        assert not (tmp_path / "workspace").exists()
