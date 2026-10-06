"""观测策略、provider 单层包装与环境资源所有权。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import iris.config as global_config
from iris.agents import AgentConfig
from iris.exceptions import IrisConfigError, IrisMCPError
from iris.message import LLMRequest
from iris.observability import AgentObservabilityConfig, ObservabilityExportConfig
from iris.observability.service import Observability
from iris.runtime import RuntimeActivationOutcome, RuntimeFactory, _assembly
from iris.tools import ToolExecutor, ToolRegistry

from ..harness.fakes import text_response
from .fakes import (
    FakeProvider,
    FakeRuntimeCommitPort,
    FakeStreamingProvider,
    MutableCancellationSignal,
    start_activation,
)
from .test_streaming import RecordingSink, _stream_events


@pytest.fixture
def telemetry() -> Iterator[tuple[Observability, InMemorySpanExporter]]:
    sdk = TracerProvider()
    exporter = InMemorySpanExporter()
    sdk.add_span_processor(SimpleSpanProcessor(exporter))
    obs = Observability.from_config(
        AgentObservabilityConfig(enabled=True),
        ObservabilityExportConfig(),
        tracer_provider=sdk,
    )
    yield obs, exporter
    sdk.shutdown()


def _config(tmp_path: Path, *, enabled: bool = False, memory: bool = False) -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "name": "observed",
            "model": "openai/test",
            "system": "instructions",
            "permissions": {"workspace": str(tmp_path)},
            "context_policy": {"enabled": False},
            "observability": {"enabled": enabled},
            "memory": {"enabled": memory},
        }
    )


@pytest.mark.asyncio
async def test_disabled_does_not_read_global_config_or_wrap_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(global_config, "_config", None)
    raw = FakeProvider([])
    runtime = RuntimeFactory.from_config(_config(tmp_path), provider=raw)
    env = runtime.environment
    assert env.provider is raw
    assert not env.observability.enabled
    assert env.owned_observability is None
    assert env.tool_bridge.tool_executor.observability is env.observability
    await env.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_enabled", [False, True])
@pytest.mark.parametrize("injected_enabled", [False, True])
@pytest.mark.parametrize("from_yaml", [False, True])
async def test_injected_strategy_wins_and_is_borrowed_by_factory_and_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    telemetry: tuple[Observability, InMemorySpanExporter],
    agent_enabled: bool,
    injected_enabled: bool,
    from_yaml: bool,
) -> None:
    monkeypatch.setattr(global_config, "_config", None)
    obs = telemetry[0] if injected_enabled else Observability()
    close = AsyncMock()
    monkeypatch.setattr(obs, "aclose", close)
    config = _config(tmp_path, enabled=agent_enabled)
    raw = FakeProvider([])
    if from_yaml:
        path = tmp_path / "agent.yaml"
        path.write_text(
            yaml.safe_dump(config.model_dump(mode="json", exclude_unset=True)), encoding="utf-8"
        )
        runtime = RuntimeFactory.from_config_path(path, provider=raw, observability=obs)
    else:
        runtime = RuntimeFactory.from_config(config, provider=raw, observability=obs)
    env = runtime.environment
    assert env.observability is obs
    assert env.owned_observability is None
    assert env.tool_bridge.tool_executor.observability is obs
    assert (env.provider is raw) is (not injected_enabled)
    await env.aclose()
    close.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_runtime_calls_generate_one_model_span(
    tmp_path: Path,
    telemetry: tuple[Observability, InMemorySpanExporter],
    streaming: bool,
) -> None:
    obs, exporter = telemetry
    response = text_response("完成")
    raw = (
        FakeStreamingProvider([_stream_events(response, stream_id="stream")])
        if streaming
        else FakeProvider([response])
    )
    runtime = RuntimeFactory.from_config(_config(tmp_path), provider=raw, observability=obs)
    activation = start_activation()
    result = await runtime.execute(
        activation,
        commits=FakeRuntimeCommitPort(activation),
        cancellation=MutableCancellationSignal(),
        stream_sink=RecordingSink() if streaming else None,
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    spans = exporter.get_finished_spans()
    [span] = [span for span in spans if span.attributes.get("gen_ai.operation.name") == "chat"]
    [prepare] = [span for span in spans if span.name == "iris.context.prepare"]
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["iris.model.outcome"] == "completed"
    assert span.attributes["iris.model.purpose"] == "main"
    assert span.attributes["iris.step.index"] == prepare.attributes["iris.step.index"] == 0
    assert prepare.end_time <= span.start_time
    assert len(raw.stream_requests if streaming else raw.requests) == 1
    await runtime.environment.aclose()


@pytest.mark.asyncio
async def test_memory_and_main_each_wrap_the_raw_provider_once(
    tmp_path: Path, telemetry: tuple[Observability, InMemorySpanExporter]
) -> None:
    obs, exporter = telemetry
    raw = FakeProvider([text_response(), text_response()])
    runtime = RuntimeFactory.from_config(
        _config(tmp_path, memory=True), provider=raw, observability=obs
    )
    memory = runtime.environment.memory_service
    assert memory is not None
    assert memory.observability is obs
    assert memory.overview_provider is not runtime.environment.provider
    request = LLMRequest(model="test")
    await runtime.environment.provider.complete(request)
    await memory.overview_provider.complete(request)
    assert len(raw.requests) == 2
    assert len(exporter.get_finished_spans()) == 2
    await runtime.environment.aclose()


def test_enabled_without_injection_requires_existing_global_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(global_config, "_config", None)
    with pytest.raises(IrisConfigError, match="init_config"):
        RuntimeFactory.from_config(_config(tmp_path, enabled=True), provider=FakeProvider([]))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "mcp", "command", "decision"])
async def test_owned_service_closes_last_even_when_existing_resource_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    telemetry: tuple[Observability, InMemorySpanExporter],
    failure: str | None,
) -> None:
    obs, _ = telemetry
    config = _config(tmp_path, enabled=True)
    export = ObservabilityExportConfig(traces_endpoint="http://localhost:5000/v1/traces")
    monkeypatch.setattr(global_config, "_config", global_config.Config(observability=export))
    construct = Mock(return_value=obs)
    monkeypatch.setattr(Observability, "from_config", construct)
    env = RuntimeFactory.from_config(config, provider=FakeProvider([])).environment
    construct.assert_called_once_with(config.observability, export)
    assert env.owned_observability is obs
    order: list[str] = []
    original = IrisMCPError("resource failed")

    def resource(name: str) -> AsyncMock:
        async def close() -> None:
            order.append(name)
            if failure == name:
                raise original

        return AsyncMock(side_effect=close)

    env.mcp_manager = SimpleNamespace(aclose=resource("mcp"))
    monkeypatch.setattr(env.command_binding.service, "aclose", resource("command"))
    env.owned_decision_client = SimpleNamespace(aclose=resource("decision"))
    close_obs = resource("observability")
    monkeypatch.setattr(obs, "aclose", close_obs)
    if failure is None:
        await env.aclose()
    else:
        with pytest.raises(IrisMCPError) as captured:
            await env.aclose()
        assert captured.value is original
    assert order == ["mcp", "command", "decision", "observability"]
    assert env.owned_observability is None
    close_obs.assert_awaited_once()


@pytest.mark.parametrize("injected", [False, True])
def test_assembly_failure_releases_only_untransferred_owned_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    telemetry: tuple[Observability, InMemorySpanExporter],
    injected: bool,
) -> None:
    obs, _ = telemetry
    monkeypatch.setattr(global_config, "_config", global_config.Config())
    monkeypatch.setattr(Observability, "from_config", Mock(return_value=obs))
    shutdown = Mock()
    monkeypatch.setattr(obs, "_shutdown", shutdown)
    original = IrisConfigError("context assembly failed")

    def fail(*args: Any, **kwargs: Any) -> None:
        raise original

    monkeypatch.setattr(_assembly, "_build_context_input", fail)
    with pytest.raises(IrisConfigError) as captured:
        RuntimeFactory.from_config(
            _config(tmp_path, enabled=True),
            provider=FakeProvider([]),
            observability=obs if injected else None,
        )
    assert captured.value is original
    assert shutdown.call_count == int(not injected)


def test_standalone_executor_defaults_to_disabled_and_borrows_explicit_service(
    telemetry: tuple[Observability, InMemorySpanExporter],
) -> None:
    assert not ToolExecutor(ToolRegistry()).observability.enabled
    obs, _ = telemetry
    assert ToolExecutor(ToolRegistry(), observability=obs).observability is obs
