"""记忆配置绑定 namespace、概览与生成阶段的当前契约。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

import iris.memory.config as memory_config
from iris.memory import (
    FileMemoryMirror,
    MemoryConfig,
    MemoryIOExecutionMode,
    MemoryOverviewConfig,
    MemoryService,
    MemoryStore,
    SQLiteMemoryStore,
    build_memory_service_from_config,
)
from iris.message import LLMRequest, LLMResponse, Msg
from iris.observability import AgentObservabilityConfig, ObservabilityExportConfig
from iris.observability.service import Observability
from iris.prompts import PromptSource
from iris.providers import CompletionProvider
from tests.harness.fakes import StaticProvider


@pytest.mark.parametrize(
    "config",
    [
        {"backend": "none"},
        {"backend": "sqlite"},
        {"mirror": {"enabled": False}},
        {"recall_mode": "on_turn"},
        {"max_query_terms": None},
        {"max_query_terms": 128},
        {"write_policy": {"mode": "sdk_only"}},
        {"orchestrator": {"enabled": True}},
        {"scope": {"collection": "default"}},
        {"search": {"use_fts": True}},
    ],
)
def test_memory_config_rejects_settings_without_behavior(config: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MemoryConfig.model_validate(config)


def test_memory_defaults_share_project() -> None:
    config = MemoryConfig()
    assert config.enabled is False
    assert config.read_namespaces == ["project"]
    assert config.write_namespace == "project"
    assert config.path == ".iris/memory/memory.db"


def test_memory_can_choose_separate_read_and_write_namespaces() -> None:
    config = MemoryConfig(
        read_namespaces=["project", "research/private"],
        write_namespace="research/private",
    )
    assert config.read_namespaces == ["project", "research/private"]
    assert config.write_namespace == "research/private"


def test_memory_overview_defaults_keep_generation_and_window_budgets_separate() -> None:
    """生成输入、生成输出和主请求采用预算分别配置。"""
    config = MemoryConfig()
    assert config.overview.input_budget_tokens == 96000
    assert config.overview.max_tokens == 4096
    assert config.overview.system_budget_ratio == 0.02
    assert not config.generation.enabled
    assert config.generation.flush_input_budget_tokens == 32000
    assert config.generation.dream_input_budget_tokens == 32000


@pytest.mark.parametrize(("enabled", "injected"), [(False, False), (False, True), (True, True)])
def test_memory_factory_resolves_disabled_and_injected_sources_without_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool, injected: bool
) -> None:
    """关闭优先于注入，开启注入也不解析配置路径或改写宿主依赖。"""
    store = Mock(spec=MemoryStore)
    provider = Mock(spec=CompletionProvider)
    prompt_source = PromptSource(tmp_path / "host-prompts")
    mirror = FileMemoryMirror(tmp_path / "host-mirror")
    overview_config = MemoryOverviewConfig(max_tokens=128)
    service = MemoryService(
        store,
        mirror=mirror,
        overview_provider=provider,
        overview_model="host-model",
        overview_config=overview_config,
        prompt_source=prompt_source,
    )
    original_observability = service.observability

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("关闭或使用注入对象时不应解析路径或新建记忆依赖")

    monkeypatch.setattr(memory_config, "resolve_memory_path", forbidden)
    monkeypatch.setattr(memory_config, "SQLiteMemoryStore", forbidden)
    monkeypatch.setattr(memory_config, "FileMemoryMirror", forbidden)
    monkeypatch.setattr(memory_config, "MemoryService", forbidden)
    result = build_memory_service_from_config(
        MemoryConfig(enabled=enabled, root="../unused", path="../unused.db"),
        tmp_path,
        memory_service=service if injected else None,
        overview_provider=Mock(spec=CompletionProvider),
        overview_model="agent-model",
        prompt_source=PromptSource(tmp_path / "agent-prompts"),
        observability=Observability(),
    )

    assert result is (service if enabled else None)
    assert service.store is store
    assert service.mirror is mirror
    assert service.overview_provider is provider
    assert service.overview_model == "host-model"
    assert service.overview_config is overview_config
    assert service.prompt_source is prompt_source
    assert service.io_execution_mode is MemoryIOExecutionMode.INLINE
    assert service.observability is original_observability
    assert store.mock_calls == []
    assert provider.mock_calls == []
    assert list(tmp_path.iterdir()) == []


def test_enabled_memory_factory_builds_sqlite_and_binds_generation_dependencies(
    tmp_path: Path,
) -> None:
    """无注入时沿唯一 SQLite 构造路径绑定依赖，构造本身不调用模型。"""
    config = MemoryConfig(
        enabled=True,
        root="memory-view",
        path="data/memory.db",
        overview=MemoryOverviewConfig(max_tokens=256),
    )
    provider = Mock(spec=CompletionProvider)
    prompt_source = PromptSource.initialize(tmp_path)
    service = build_memory_service_from_config(
        config,
        tmp_path,
        overview_provider=provider,
        overview_model="agent-model",
        prompt_source=prompt_source,
    )

    assert service is not None
    assert isinstance(service.store, SQLiteMemoryStore)
    assert service.store.path == tmp_path / "data/memory.db"
    assert service.store.path.is_file()
    assert service.mirror is not None
    assert service.mirror.root == tmp_path / "memory-view"
    assert service.overview_provider is provider
    assert service.overview_model == "agent-model"
    assert service.overview_config is config.overview
    assert service.generation_provider is provider
    assert service.generation_model == "agent-model"
    assert service.generation_config is config.generation
    assert service.prompt_source is prompt_source
    assert service.io_execution_mode is MemoryIOExecutionMode.THREAD
    assert provider.mock_calls == []
    assert list(service.mirror.root.rglob("Memory.md")) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("from_config", [False, True])
async def test_memory_wraps_each_raw_provider_once_with_borrowed_observability(
    tmp_path: Path, from_config: bool
) -> None:
    """工厂仅传递依赖，构造器为实际持有的两个模型入口各包一层。"""
    sdk = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    sdk.add_span_processor(SimpleSpanProcessor(exporter))
    observability = Observability.from_config(
        AgentObservabilityConfig(enabled=True), ObservabilityExportConfig(), tracer_provider=sdk
    )
    raw = StaticProvider(LLMResponse(provider="test"), LLMResponse(provider="test"))
    source = PromptSource.initialize(tmp_path)
    if from_config:
        service = build_memory_service_from_config(
            MemoryConfig(enabled=True),
            tmp_path,
            overview_provider=raw,
            overview_model="test",
            prompt_source=source,
            observability=observability,
        )
    else:
        service = MemoryService(
            Mock(spec=MemoryStore),
            overview_provider=raw,
            generation_provider=raw,
            observability=observability,
        )
    assert service is not None
    assert service.observability is observability
    request = LLMRequest(model="test", messages=[Msg.user("inspect")])
    try:
        await service.overview_provider.complete(request)
        await service.generation_provider.complete(request)
        assert raw.requests == [request, request]
        assert len(exporter.get_finished_spans()) == 2
        await observability.aclose()
        with sdk.get_tracer("host").start_as_current_span("still-open"):
            pass
        assert len(exporter.get_finished_spans()) == 3
    finally:
        sdk.shutdown()
