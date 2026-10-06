"""CLI 宿主显式装配维护资源，并在退出时收口后台生成。"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import iris.config as iris_config
from iris.cli import chat
from iris.exceptions import IrisConfigError
from iris.message import LLMRequest, LLMResponse
from iris.observability import AgentObservabilityConfig
from iris.observability.service import Observability
from tests.cli.test_chat_streaming import _text_stream
from tests.harness.fakes import StaticProvider
from tests.runtime.fakes import FakeStreamingProvider


class ChatMaintenanceProvider(FakeStreamingProvider):
    """前台立即完成，后台生成等待宿主取消。"""

    def __init__(self) -> None:
        super().__init__([_text_stream()])
        self.generating = threading.Event()
        self.cancelled = threading.Event()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """前台使用 stream，后台 complete 等待取消。"""
        self._requests.append(request)
        self.generating.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()
        raise AssertionError("测试中的后台生成只能被取消")


def _host_observability(
    monkeypatch: pytest.MonkeyPatch, on_shutdown: Callable[[], None]
) -> tuple[Observability, InMemorySpanExporter]:
    """用实际 SDK 验证宿主最后关闭，不需要 OTLP 服务。"""
    sdk = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    sdk.add_span_processor(SimpleSpanProcessor(exporter))
    observability = Observability(AgentObservabilityConfig(enabled=True), sdk, owned_provider=sdk)
    original_shutdown = sdk.shutdown

    def shutdown() -> None:
        on_shutdown()
        original_shutdown()

    monkeypatch.setattr(sdk, "shutdown", shutdown)
    monkeypatch.setattr(Observability, "from_config", Mock(return_value=observability))
    monkeypatch.setattr(iris_config, "_config", iris_config.Config())
    return observability, exporter


@pytest.mark.parametrize("feature", ["memory", "evolution"])
def test_chat_owns_maintenance_and_drains_before_loop_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, feature: str
) -> None:
    """实际 CLI 路径应完成绑定、空闲生成与退出取消，不要求 SDK 用户代劳。"""
    config_path = tmp_path / "agent.yaml"
    feature_config = (
        "memory:\n  enabled: true\n  generation:\n    enabled: true\n"
        if feature == "memory"
        else "skills:\n  enabled: true\nevolution:\n  enabled: true\n"
    )
    config_path.write_text(
        "name: chat\nmodel: openai/test\nsystem: help\n"
        "observability:\n  enabled: true\n"
        "maintenance:\n  idle_seconds: 0\n" + feature_config,
        encoding="utf-8",
    )
    provider = ChatMaintenanceProvider()
    shutdowns: list[str] = []

    def shutdown() -> None:
        assert provider.cancelled.is_set()
        shutdowns.append("exporter")

    observability, exporter = _host_observability(monkeypatch, shutdown)
    captured: list[chat.AgentRunner] = []
    original_loop = chat.run_chat_loop

    def run_loop(*, runner: chat.AgentRunner, **kwargs: object) -> int:
        captured.append(runner)
        assert kwargs["observability"] is observability
        assert runner.runtime.environment.observability is observability
        assert kwargs["maintenance"].observability is observability
        if feature == "memory":
            assert runner.runtime.environment.memory_service.observability is observability
        else:
            assert runner._maintenance.project.binding.service.observability is observability
        return original_loop(runner=runner, **kwargs)

    monkeypatch.setattr(chat, "run_chat_loop", run_loop)
    monkeypatch.setattr(chat, "is_config_initialized", lambda: True)
    monkeypatch.setattr(chat, "create_provider_client", lambda *args, **kwargs: provider)
    inputs = iter(["开始", "/exit"])
    errors: list[str] = []

    def read_input(prompt: str) -> str:
        value = next(inputs)
        if value == "/exit":
            assert provider.generating.wait(5)
        return value

    assert (
        chat.run_chat(
            chat.ChatOptions(config_path=config_path),
            input_func=read_input,
            output_func=lambda text: None,
            error_func=errors.append,
        )
        == 0
    )
    assert errors == []
    assert provider.cancelled.is_set()
    assert len(provider.stream_requests) == len(provider.requests) == 1
    assert len(captured) == 1
    assert shutdowns == ["exporter"]
    assert len(exporter.get_finished_spans()) == 2
    if feature == "evolution":
        assert not (tmp_path / ".iris" / "memory").exists()


def test_chat_preparation_failure_reports_error_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """维护准备失败仍沿 CLI 错误出口返回，并按宿主所有权收尾。"""
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "name: chat\nmodel: openai/test\nsystem: help\n"
        "observability:\n  enabled: true\n"
        "memory:\n  enabled: true\n  generation:\n    enabled: true\n",
        encoding="utf-8",
    )
    timeline: list[str] = []
    _host_observability(monkeypatch, lambda: timeline.append("exporter"))
    errors: list[str] = []
    original_runner_close = chat.AgentRunner.aclose
    original_maintenance_close = chat.MaintenanceCoordinator.aclose

    async def fail_prepare(self: chat.AgentRunner) -> None:
        timeline.append("prepare")
        raise IrisConfigError("测试准备失败")

    async def close_runner(self: chat.AgentRunner) -> None:
        await original_runner_close(self)
        timeline.append("runner")

    async def close_maintenance(self: chat.MaintenanceCoordinator) -> None:
        await original_maintenance_close(self)
        timeline.append("maintenance")

    monkeypatch.setattr(chat, "is_config_initialized", lambda: True)
    monkeypatch.setattr(chat, "create_provider_client", lambda *args, **kwargs: StaticProvider())
    monkeypatch.setattr(chat.AgentRunner, "aprepare", fail_prepare)
    monkeypatch.setattr(chat.AgentRunner, "aclose", close_runner)
    monkeypatch.setattr(chat.MaintenanceCoordinator, "aclose", close_maintenance)

    assert (
        chat.run_chat(
            chat.ChatOptions(config_path=config_path),
            input_func=lambda prompt: pytest.fail("准备失败后不应接收输入"),
            output_func=lambda text: None,
            error_func=errors.append,
        )
        == 1
    )
    assert len(errors) == 1 and "测试准备失败" in errors[0]
    assert timeline == ["prepare", "maintenance", "runner", "exporter"]


@pytest.mark.parametrize("failure_at", ["memory", "evolution", "runner"])
@pytest.mark.parametrize("domain_error", [False, True])
def test_chat_construction_failure_closes_untransferred_observability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str, domain_error: bool
) -> None:
    """后台 loop 接手前的同步构造失败也只能关闭一次自有 SDK。"""
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "name: chat\nmodel: openai/test\nsystem: help\nobservability:\n  enabled: true\n",
        encoding="utf-8",
    )
    shutdowns: list[str] = []
    _host_observability(monkeypatch, lambda: shutdowns.append("exporter"))
    monkeypatch.setattr(chat, "create_provider_client", lambda *args, **kwargs: StaticProvider())
    failure = IrisConfigError("构造失败") if domain_error else RuntimeError("构造失败")
    construct = Mock(side_effect=failure)
    if failure_at == "memory":
        monkeypatch.setattr(chat, "build_memory_service_from_config", construct)
    elif failure_at == "evolution":
        monkeypatch.setattr(chat, "build_project_evolution_binding", construct)
    else:
        monkeypatch.setattr(chat.AgentRunner, "from_config", construct)
    errors: list[str] = []

    def run() -> int:
        return chat.run_chat(
            chat.ChatOptions(config_path=config_path),
            input_func=lambda prompt: pytest.fail("同步构造失败不能启动输入循环"),
            output_func=lambda text: None,
            error_func=errors.append,
        )

    if domain_error:
        assert run() == 1
        assert len(errors) == 1 and "构造失败" in errors[0]
    else:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is failure
    construct.assert_called_once()
    assert shutdowns == ["exporter"]


def test_chat_loop_keeps_runner_observability_borrowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """直接传 runner 不等于将其宿主共享观测资源交给 chat loop。"""
    shutdowns: list[str] = []
    observability, _ = _host_observability(monkeypatch, lambda: shutdowns.append("exporter"))
    config = chat.load_agent_config
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("name: chat\nmodel: openai/test\nsystem: help\n", encoding="utf-8")
    runner = chat.AgentRunner.from_config(
        config(config_path),
        config_path=config_path,
        provider=StaticProvider(),
        observability=observability,
    )
    try:
        assert (
            chat.run_chat_loop(
                runner=runner,
                options=chat.ChatOptions(config_path=config_path),
                input_func=lambda prompt: "/exit",
                output_func=lambda text: None,
                error_func=lambda text: pytest.fail(text),
            )
            == 0
        )
        assert shutdowns == []
    finally:
        observability._shutdown()
    assert shutdowns == ["exporter"]


@pytest.mark.parametrize("failure_at", ["host", "manager"])
def test_direct_chat_loop_closes_explicit_observability_before_session_is_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    """显式传给 loop 的资源也覆盖 host 同步构造与后台 manager 构造失败。"""
    shutdowns: list[str] = []
    observability, _ = _host_observability(monkeypatch, lambda: shutdowns.append("exporter"))
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("name: chat\nmodel: openai/test\nsystem: help\n", encoding="utf-8")
    runner = chat.AgentRunner.from_config(
        chat.load_agent_config(config_path),
        config_path=config_path,
        provider=StaticProvider(),
        observability=observability,
    )
    failure = RuntimeError("会话构造失败")
    monkeypatch.setattr(
        chat,
        "_ChatSessionHost" if failure_at == "host" else "SessionManager",
        Mock(side_effect=failure),
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            chat.run_chat_loop(
                runner=runner,
                options=chat.ChatOptions(config_path=config_path),
                observability=observability,
                input_func=lambda prompt: pytest.fail("构造失败不能读取输入"),
                output_func=lambda text: None,
                error_func=lambda text: pytest.fail(text),
            )
        assert caught.value is failure
        assert shutdowns == ["exporter"]
    finally:
        asyncio.run(runner.aclose())
        observability._shutdown()
