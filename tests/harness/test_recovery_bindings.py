"""Memory 提前绑定与 Evolution 晚绑定使用实际资源路径。"""

from pathlib import Path

import pytest

from iris.exceptions import IrisConfigError
from iris.harness._recovery_bindings import (
    describe_configured_memory,
    describe_memory_state,
    validate_memory_restore,
)
from iris.memory import MemoryConfig, MemoryService, SQLiteMemoryStore
from iris.memory.mirror import FileMemoryMirror

from ..command.test_docker import FakeClient, driver

__all__ = ["driver"]


@pytest.mark.parametrize(
    "field,value",
    [("path", ".iris/other.db"), ("root", ".iris/other"), ("write_namespace", "other")],
)
def test_configured_memory_mismatch_does_not_create_files(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    config = MemoryConfig(enabled=True)
    expected = describe_configured_memory(config, workspace=tmp_path)
    with pytest.raises(IrisConfigError):
        validate_memory_restore(
            config.model_copy(update={field: value}), None, workspace=tmp_path, expected=expected
        )
    assert not list(tmp_path.iterdir())


def test_injected_memory_uses_actual_paths_and_disabled_ignores_injection(tmp_path: Path) -> None:
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / "custom.db"), mirror=FileMemoryMirror(tmp_path / "projection")
    )
    expected = describe_memory_state(service, workspace=tmp_path, namespace="project")
    validate_memory_restore(
        MemoryConfig(enabled=True), service, workspace=tmp_path, expected=expected
    )
    with pytest.raises(IrisConfigError):
        validate_memory_restore(
            MemoryConfig(enabled=True), None, workspace=tmp_path, expected=expected
        )
    validate_memory_restore(MemoryConfig(enabled=False), service, workspace=tmp_path, expected=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("injected", [False, True])
async def test_restore_state_watermarks_and_late_bindings(
    tmp_path: Path,
    driver: FakeClient,
    injected: bool,
) -> None:
    from iris.agents import AgentSkillsConfig
    from iris.evolution.config import EvolutionConfig
    from iris.harness import (
        AgentRunner,
        MaintenanceCoordinator,
        MemoryMaintenanceBinding,
        build_project_evolution_binding,
    )
    from iris.memory.generation_models import MemoryCaptureSource
    from iris.prompts import PromptSource

    from ..evolution.test_materials import _block, _source
    from .fakes import StaticProvider
    from .test_environment_restore import vacate
    from .test_recovery_point import snapshot_runner

    base = snapshot_runner(tmp_path)
    workspace = tmp_path / "workspace"
    config = base.runtime.environment.agent_config.model_copy(
        update={
            "memory": MemoryConfig(enabled=True),
            "skills": AgentSkillsConfig(enabled=True),
            "evolution": EvolutionConfig(
                enabled=True, prompt_targets=("compaction",), config_targets=("system",)
            ),
        }
    )
    await base.aclose()
    runner = AgentRunner.from_config(config, provider=StaticProvider())
    memory = runner.runtime.environment.memory_service
    namespace = config.memory.write_namespace
    source = memory.store.register_source(
        MemoryCaptureSource(
            lifecycle_source_id=runner.store.source_id,
            run_id="material",
            session_id="s",
            namespace=namespace,
            initial_message_count=2,
            captured_until=4,
        )
    )
    config_path = workspace / "agent.yaml"
    config_path.write_text("system: test")
    evolution = build_project_evolution_binding(
        config,
        workspace_root=workspace,
        prompt_source=PromptSource(workspace / ".iris/prompts"),
        provider=StaticProvider(),
        config_path=config_path,
    )
    project_source = _source()
    evolution.service.store.register_source(project_source, 2)
    evolution.service.store.commit_capture(_block(project_source, 2, 4))
    coordinator = MaintenanceCoordinator(idle_seconds=9999)
    runner.bind_maintenance(
        coordinator,
        memory=MemoryMaintenanceBinding(
            service=memory,
            database_path=memory.store.path,
            namespace=namespace,
        ),
        evolution=evolution,
    )
    directory = tmp_path / "snapshots"
    saved = await runner.save_environment_snapshot(directory)
    assert {state.role for state in saved.states} == {"memory", "evolution"}
    memory.store.commit_capture(
        source.model_copy(update={"captured_until": 5}), expected_captured_until=4, episode=None
    )
    evolution.service.store.commit_capture(_block(project_source, 4, 5))
    await runner.aclose()
    await coordinator.aclose()
    vacate(runner, tmp_path)
    driver.closed = False
    restored = await AgentRunner.from_environment_snapshot(
        config,
        directory,
        provider=StaticProvider(),
        memory_service=memory if injected else None,
    )
    actual_memory = restored.runtime.environment.memory_service
    if injected:
        assert actual_memory is memory
    assert (
        actual_memory.store.list_capture_sources(saved.source_id, namespace)[0].captured_until == 4
    )
    restored_evolution = build_project_evolution_binding(
        config,
        workspace_root=workspace,
        prompt_source=PromptSource(workspace / ".iris/prompts"),
        provider=StaticProvider(),
        config_path=config_path,
    )
    assert restored_evolution.service.store.list_capture_sources("lifecycle")[0].captured_until == 4
    coordinator = MaintenanceCoordinator(idle_seconds=9999)
    good_memory = MemoryMaintenanceBinding(
        service=actual_memory, database_path=actual_memory.store.path, namespace=namespace
    )
    with pytest.raises(IrisConfigError, match="Memory"):
        restored.bind_maintenance(
            coordinator,
            memory=MemoryMaintenanceBinding(
                service=actual_memory,
                database_path=workspace / "wrong.db",
                namespace=namespace,
            ),
        )
    assert not coordinator._readers and restored._capture is None
    original = restored_evolution.service.skill_path
    restored_evolution.service.skill_path = workspace / "other/SKILL.md"
    with pytest.raises(IrisConfigError, match="Evolution"):
        restored.bind_maintenance(coordinator, memory=good_memory, evolution=restored_evolution)
    assert not coordinator._readers and restored._capture is None
    restored_evolution.service.skill_path = original
    restored.bind_maintenance(coordinator, memory=good_memory, evolution=restored_evolution)
    assert restored._capture is not None
    await restored.aclose()
    await coordinator.aclose()


@pytest.mark.asyncio
async def test_real_capture_watermarks_rewind_and_continue(
    tmp_path: Path, driver: FakeClient
) -> None:
    from iris.agents import AgentSkillsConfig, ToolsConfig
    from iris.evolution.config import EvolutionConfig
    from iris.harness import (
        AgentRunner,
        MaintenanceCoordinator,
        MemoryMaintenanceBinding,
        build_project_evolution_binding,
    )
    from iris.hitl import QuestionInteractionResponse
    from iris.lifecycle import AgentRunRequest
    from iris.message import ToolUseBlock
    from iris.prompts import PromptSource

    from .fakes import StaticProvider, text_response, tool_response
    from .test_environment_restore import vacate
    from .test_recovery_point import snapshot_runner

    base = snapshot_runner(tmp_path)
    workspace = tmp_path / "workspace"
    config = base.runtime.environment.agent_config.model_copy(
        update={
            "memory": MemoryConfig(enabled=True),
            "skills": AgentSkillsConfig(enabled=True),
            "evolution": EvolutionConfig(enabled=True),
            "tools": ToolsConfig(builtin=["human.ask"]),
        }
    )
    await base.aclose()
    runner = AgentRunner.from_config(
        config,
        provider=StaticProvider(
            tool_response(
                ToolUseBlock(id="ask", name="ask_question", input={"question": "继续？"})
            ),
            text_response("advanced"),
        ),
    )

    def bind(target: AgentRunner) -> tuple[MaintenanceCoordinator, object]:
        memory = target.runtime.environment.memory_service
        evolution = build_project_evolution_binding(
            config,
            workspace_root=workspace,
            prompt_source=PromptSource(workspace / ".iris/prompts"),
            provider=StaticProvider(),
        )
        coordinator = MaintenanceCoordinator(idle_seconds=9999)
        target.bind_maintenance(
            coordinator,
            memory=MemoryMaintenanceBinding(
                service=memory, database_path=memory.store.path, namespace="project"
            ),
            evolution=evolution,
        )
        return coordinator, evolution.service

    coordinator, evolution = bind(runner)
    waiting = await runner.start(AgentRunRequest(input="ask", session_id="s"))
    directory = tmp_path / "snapshots"
    saved = await runner.save_environment_snapshot(directory)
    memory_before = runner.runtime.environment.memory_service.store.list_capture_sources(
        saved.source_id, "project"
    )
    evolution_before = evolution.store.list_capture_sources(saved.source_id)
    assert len(memory_before) == len(evolution_before) == 1
    assert memory_before[0].captured_until > 0
    await runner.resume(
        waiting.run.run_id,
        interaction_id=waiting.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="advance"),
    )
    assert (
        runner.runtime.environment.memory_service.store.list_capture_sources(
            saved.source_id, "project"
        )
        == []
    )
    assert evolution.store.list_capture_sources(saved.source_id) == ()
    await runner.aclose()
    await coordinator.aclose()
    vacate(runner, tmp_path)
    driver.closed = False
    restored = await AgentRunner.from_environment_snapshot(
        config, directory, provider=StaticProvider(text_response("restored"))
    )
    assert (
        restored.runtime.environment.memory_service.store.list_capture_sources(
            saved.source_id, "project"
        )
        == memory_before
    )
    coordinator, evolution = bind(restored)
    assert evolution.store.list_capture_sources(saved.source_id) == evolution_before
    await restored.resume(
        waiting.run.run_id,
        interaction_id=waiting.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="continue"),
    )
    assert (
        restored.runtime.environment.memory_service.store.list_capture_sources(
            saved.source_id, "project"
        )
        == []
    )
    assert evolution.store.list_capture_sources(saved.source_id) == ()
    await restored.aclose()
    await coordinator.aclose()


@pytest.mark.asyncio
async def test_shared_database_namespaces_are_not_exclusive_to_root(
    tmp_path: Path,
    driver: FakeClient,
) -> None:
    from iris.agents import SessionConfig
    from iris.harness import AgentRunner, MaintenanceCoordinator, MemoryMaintenanceBinding

    from .fakes import StaticProvider
    from .test_recovery_point import snapshot_runner

    first = snapshot_runner(tmp_path / "first", memory=True)
    memory = first.runtime.environment.memory_service
    config = first.runtime.environment.agent_config.model_copy(
        update={
            "session": SessionConfig(backend="sqlite", path=str(tmp_path / "second.sqlite")),
            "memory": MemoryConfig(enabled=True, write_namespace="second"),
        }
    )
    second = AgentRunner.from_config(config, provider=StaticProvider(), memory_service=memory)
    coordinator = MaintenanceCoordinator(idle_seconds=9999)
    for runner, namespace in ((first, "project"), (second, "second")):
        runner.bind_maintenance(
            coordinator,
            memory=MemoryMaintenanceBinding(
                service=memory, database_path=memory.store.path, namespace=namespace
            ),
        )
    try:
        with pytest.raises(IrisConfigError, match="独占"):
            await first.save_environment_snapshot(tmp_path / "snapshots")
        assert not driver.info_calls
    finally:
        await first.aclose()
        await second.aclose()
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_shared_coordinator_does_not_wait_for_other_root_io(
    tmp_path: Path,
    driver: FakeClient,
) -> None:
    import asyncio
    import threading

    from iris.harness import MaintenanceCoordinator, MemoryMaintenanceBinding

    from .test_recovery_point import snapshot_runner

    first = snapshot_runner(tmp_path / "first", memory=True)
    second = snapshot_runner(tmp_path / "second", memory=True)
    coordinator = MaintenanceCoordinator(idle_seconds=9999)
    for runner in (first, second):
        memory = runner.runtime.environment.memory_service
        runner.bind_maintenance(
            coordinator,
            memory=MemoryMaintenanceBinding(
                service=memory, database_path=memory.store.path, namespace="project"
            ),
        )
    entered, release = threading.Event(), threading.Event()

    def other_capture() -> None:
        entered.set()
        assert release.wait(8)

    task = asyncio.create_task(
        second.runtime.environment.memory_service.run_async_io(other_capture)
    )
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        async with asyncio.timeout(5):
            await first.save_environment_snapshot(tmp_path / "snapshots")
        assert not task.done()
    finally:
        release.set()
        await task
        await first.aclose()
        await second.aclose()
        await coordinator.aclose()
