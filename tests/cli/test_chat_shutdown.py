"""CLI 退出和 terminal 展示延迟的真实 host 回归。"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Literal

import pytest

from iris.cli.chat import ChatOptions, _ChatSessionHost, run_chat_loop
from iris.harness import AgentRunner, RunPhase, RunStopReason
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry
from tests.harness.fakes import StaticProvider, build_runtime, text_response, tool_response


@pytest.mark.parametrize("exit_kind", ["command", "eof", "interrupt"])
def test_waiting_follow_up_is_not_started_during_cli_exit(tmp_path: Path, exit_kind: str) -> None:
    """退出时 pending follow-up 失败，当前 waiting run 完成 durable cancellation。"""
    registry = ToolRegistry()
    registry.register_function(
        lambda: "写入", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="write", name="write", input={})), text_response("意外启动")
    )
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider, registry=registry), store=store
    )
    prompted = threading.Event()
    current: list[str] = []
    errors: list[str] = []
    index = 0

    def read_input(prompt: str) -> str:
        nonlocal index
        index += 1
        if index == 1:
            return "开始"
        if index == 2:
            assert prompted.wait(2)
            current.append(store.load_session_lane("cli"))
            return "/follow-up 以后再执行"
        if exit_kind == "eof":
            raise EOFError
        if exit_kind == "interrupt":
            raise KeyboardInterrupt
        return "/exit"

    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=read_input,
        output_func=lambda message: prompted.set() if message == "批准该调用？ [y/N]" else None,
        error_func=errors.append,
    )
    assert code == (130 if exit_kind == "interrupt" else 0)
    assert errors == []
    assert len(provider.requests) == 1
    assert runner.get_run(current[0]).stop_reason is RunStopReason.CANCELLED
    assert store.load_session_lane("cli") is None


def test_plain_text_routes_while_previous_terminal_display_is_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durable terminal 已完成，但 consumer 尚未显示时，下一行仍能开新 run。"""
    provider = StaticProvider(text_response("第一轮"), text_response("第二轮"))
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    terminal_pending = threading.Event()
    release_display = threading.Event()
    finished = threading.Event()
    errors: list[str] = []
    original_submit = _ChatSessionHost._submit

    async def delay_display(self: _ChatSessionHost, run_id: str) -> None:
        terminal_pending.set()
        assert await asyncio.to_thread(release_display.wait, 2)

    async def submit_and_release(
        self: _ChatSessionHost, input: str, *, mode: Literal["follow_up"] | None = None
    ) -> None:
        try:
            await original_submit(self, input, mode=mode)
        finally:
            if input == "继续":
                release_display.set()

    monkeypatch.setattr(_ChatSessionHost, "_wait_for_live_terminal", delay_display)
    monkeypatch.setattr(_ChatSessionHost, "_submit", submit_and_release)
    index = 0

    def read_input(prompt: str) -> str:
        nonlocal index
        index += 1
        if index == 1:
            return "开始"
        if index == 2:
            assert terminal_pending.wait(2)
            return "继续"
        assert finished.wait(2)
        return "/exit"

    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=read_input,
        output_func=lambda message: finished.set() if message == "第二轮" else None,
        error_func=errors.append,
    )
    assert code == 0
    assert errors == []
    assert [message.text for message in store.load_session("cli").messages] == [
        "开始",
        "第一轮",
        "继续",
        "第二轮",
    ]


def test_exit_before_waiting_prompt_consumption_ignores_stale_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """关闭先完成 cancellation 时，迟到的 suspended event 不再是当前输入提示。"""
    registry = ToolRegistry()
    registry.register_function(
        lambda: "写入", name="write", description="写入", capabilities={ToolCapability.WRITE}
    )
    provider = StaticProvider(tool_response(ToolUseBlock(id="write", name="write", input={})))
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider, registry=registry), store=store
    )
    original_consume = _ChatSessionHost._consume_events
    run_ids: list[str] = []

    async def delayed_consume(self: _ChatSessionHost) -> None:
        await self._stop.wait()
        while runner.get_run(run_ids[0]).phase is not RunPhase.TERMINAL:
            await asyncio.sleep(0)
        await original_consume(self)

    monkeypatch.setattr(_ChatSessionHost, "_consume_events", delayed_consume)
    outputs: list[str] = []
    host = _ChatSessionHost(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        live_broker=None,
        output_func=outputs.append,
        stream_output_func=outputs.append,
        error_func=outputs.append,
    )
    host.start()
    host.submit("开始")

    async def wait_for_waiting() -> None:
        run_id = store.load_session_lane("cli")
        run_ids.append(run_id)
        while runner.get_run(run_id).phase is not RunPhase.WAITING:
            await asyncio.sleep(0)

    host._call(asyncio.wait_for(wait_for_waiting(), timeout=2))
    host.close(reason="用户退出")
    assert runner.get_run(run_ids[0]).stop_reason is RunStopReason.CANCELLED
    assert "批准该调用？ [y/N]" not in outputs
