"""CLI 宿主显式装配维护资源，并在退出时收口后台生成。"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from iris.cli import chat
from iris.exceptions import IrisConfigError
from iris.message import LLMRequest, LLMResponse
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


def test_chat_owns_maintenance_and_drains_before_loop_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """实际 CLI 路径应完成绑定、空闲生成与退出取消，不要求 SDK 用户代劳。"""
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "name: chat\nmodel: openai/test\nsystem: help\n"
        "maintenance:\n  idle_seconds: 0\n"
        "memory:\n  enabled: true\n  generation:\n    enabled: true\n",
        encoding="utf-8",
    )
    provider = ChatMaintenanceProvider()
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


def test_chat_preparation_failure_reports_error_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """维护准备失败仍沿 CLI 错误出口返回，并按宿主所有权收尾。"""
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "name: chat\nmodel: openai/test\nsystem: help\n"
        "memory:\n  enabled: true\n  generation:\n    enabled: true\n",
        encoding="utf-8",
    )
    timeline: list[str] = []
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
    assert timeline == ["prepare", "maintenance", "runner"]
