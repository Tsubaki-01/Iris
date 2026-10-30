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
