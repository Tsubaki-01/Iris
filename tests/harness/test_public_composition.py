"""Harness-owned lifecycle composition 与 runtime dependency direction 测试。"""

from __future__ import annotations

from pathlib import Path

from iris.agents import AgentConfig
from iris.harness import AgentRunner
from iris.runtime import RuntimeFactory
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .fakes import StaticProvider, text_response


def _config(*, backend: str = "none", path: str | None = None) -> AgentConfig:
    session: dict[str, str] = {"backend": backend}
    if path is not None:
        session["path"] = path
    return AgentConfig(
        name="composition-agent",
        model={"provider": "openai", "name": "fake-model"},
        system="测试",
        session=session,
    )


def test_explicit_store_identity_is_preserved_for_every_runner_view() -> None:
    store = InMemoryLifecycleStore()

    runner = AgentRunner.from_config(
        _config(),
        provider=StaticProvider(text_response()),
        store=store,
    )

    assert runner.store is store
    assert not hasattr(runner.runtime.environment, "session_store")
    assert not hasattr(runner.runtime.environment, "interaction_service")


def test_falsey_explicit_store_identity_is_preserved(tmp_path: Path) -> None:
    class FalseyStore(InMemoryLifecycleStore):
        def __bool__(self) -> bool:
            return False

    store = FalseyStore()
    database = tmp_path / "must-not-be-created.db"

    runner = AgentRunner.from_config(
        _config(backend="sqlite", path=str(database)),
        provider=StaticProvider(text_response()),
        store=store,
    )

    assert runner.store is store
    assert not database.exists()


def test_backend_none_selects_in_memory_lifecycle_store() -> None:
    runner = AgentRunner.from_config(
        _config(),
        provider=StaticProvider(text_response()),
    )

    assert isinstance(runner.store, InMemoryLifecycleStore)


def test_sqlite_store_path_is_relative_to_agent_config(tmp_path: Path) -> None:
    config_path = tmp_path / "agents" / "agent.yaml"
    config_path.parent.mkdir()

    runner = AgentRunner.from_config(
        _config(backend="sqlite", path="state/lifecycle.db"),
        config_path=config_path,
        provider=StaticProvider(text_response()),
    )

    assert isinstance(runner.store, SQLiteStore)
    assert runner.store.path == (config_path.parent / "state" / "lifecycle.db").resolve()
    assert runner.store.path.is_file()


def test_runtime_factory_alone_never_creates_lifecycle_database(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    database = tmp_path / "state" / "lifecycle.db"

    runtime = RuntimeFactory.from_config(
        _config(backend="sqlite", path="state/lifecycle.db"),
        config_path=config_path,
        provider=StaticProvider(text_response()),
    )

    assert runtime.environment.agent_config.name == "composition-agent"
    assert not database.exists()
