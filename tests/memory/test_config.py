from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from iris.exceptions import IrisConfigError
from iris.memory import (
    MemoryBackend,
    MemoryConfig,
    MemoryMirrorMode,
    MemoryService,
    MemoryVisibility,
    build_memory_service_from_config,
    resolve_memory_path,
)


def test_memory_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MemoryConfig(backend="none", embedding={"provider": "none"})

    with pytest.raises(ValidationError):
        MemoryConfig(search={"limit": 5, "reranker": "none"})


def test_memory_config_defaults_are_conservative() -> None:
    config = MemoryConfig()

    assert config.backend == MemoryBackend.NONE
    assert config.root == ".iris/memory"
    assert config.path == ".iris/memory/memory.db"
    assert config.scope.collection == "default"
    assert config.scope.visibility == MemoryVisibility.AGENT
    assert config.search.limit == 10
    assert config.search.use_fts is True
    assert config.mirror.enabled is True
    assert config.mirror.mode == MemoryMirrorMode.MINIMAL
    assert config.write_policy.mode == "sdk_only"
    assert config.write_policy.delete_mode == "tombstone"
    assert config.orchestrator.enabled is False


def test_backend_none_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    service = build_memory_service_from_config(MemoryConfig(backend="none"), tmp_path)

    assert service is None
    assert not (tmp_path / ".iris").exists()


def test_backend_sqlite_creates_service_db_and_mirror_layout(tmp_path: Path) -> None:
    service = build_memory_service_from_config(MemoryConfig(backend="sqlite"), tmp_path)

    assert isinstance(service, MemoryService)
    assert (tmp_path / ".iris" / "memory" / "memory.db").exists()
    assert (tmp_path / ".iris" / "memory" / "Memory.md").exists()


def test_backend_sqlite_can_disable_mirror_layout(tmp_path: Path) -> None:
    config = MemoryConfig(backend="sqlite", mirror={"enabled": False})

    service = build_memory_service_from_config(config, tmp_path)

    assert isinstance(service, MemoryService)
    assert (tmp_path / ".iris" / "memory" / "memory.db").exists()
    assert not (tmp_path / ".iris" / "memory" / "Memory.md").exists()


def test_resolve_memory_path_rejects_workspace_escape(tmp_path: Path) -> None:
    with pytest.raises(IrisConfigError):
        resolve_memory_path("../outside.db", tmp_path)


def test_scope_config_builds_scope_from_runtime_values() -> None:
    config = MemoryConfig(scope={"collection": "notes"})

    scope = config.scope.to_scope(workspace_id="workspace", agent_id="agent")

    assert scope.workspace_id == "workspace"
    assert scope.agent_id == "agent"
    assert scope.collection == "notes"


def test_session_scope_requires_runtime_session_id() -> None:
    config = MemoryConfig(scope={"visibility": "session"})

    with pytest.raises(IrisConfigError):
        config.scope.to_scope(workspace_id="workspace", agent_id="agent")

    scope = config.scope.to_scope(
        workspace_id="workspace",
        agent_id="agent",
        session_id="session",
    )
    assert scope.session_id == "session"
