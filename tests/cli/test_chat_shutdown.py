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
from tests.harness.test_runner_subagent import ChildProviders, _parent_provider
from tests.harness.test_runner_subagent_mcp import configs
from tests.mcp.fixtures.runtime import MCPPeer, mcp_agent


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


@pytest.mark.parametrize("waiting_parent", [False, True])
def test_mcp_shutdown_drains_original_calls_before_closing_background_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    waiting_parent: bool,
) -> None:
    peer = MCPPeer(monkeypatch)
    peer.block_call = True
    peer.release_cleanup.clear()
    mcp_response = tool_response(ToolUseBlock(id="mcp-call", name="mcp__test__echo", input={}))
    if waiting_parent:
        runner = AgentRunner.from_config_path(
            configs(tmp_path, root_mcp=True),
            provider=_parent_provider(),
            child_provider_factory=ChildProviders(
                StaticProvider(
                    tool_response(
                        ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
                    ),
                    mcp_response,
                )
            ),
        )
    else:
        runner = AgentRunner.from_config(mcp_agent(tmp_path), provider=StaticProvider(mcp_response))
    errors: list[str] = []
    host = _ChatSessionHost(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        live_output=None,
        output_func=lambda text: None,
        error_func=errors.append,
    )
    host.start()
    host.submit("start")

    async def wait_for_prompt() -> None:
        async with asyncio.timeout(2):
            while host._pending_interaction is None:
                await asyncio.sleep(0)

    if waiting_parent:
        host._call(wait_for_prompt())
        host.submit("continue")
    host._call(asyncio.wait_for(peer.called.wait(), 2))
    closer = threading.Thread(target=host.close)
    closer.start()
    try:
        host._call(asyncio.wait_for(peer.cleaning.wait(), 2))
        assert closer.is_alive() and host._thread.is_alive()
        assert peer.events.count("close") == int(waiting_parent)
        active = runner.store.load_session_lane("cli")
        assert active is not None
    finally:
        host._require_loop().call_soon_threadsafe(peer.release_cleanup.set)
        closer.join(5)
    assert not closer.is_alive() and not host._thread.is_alive()
    assert peer.events.count("close") == peer.events.count("open") == 1 + 2 * int(waiting_parent)


def test_mcp_close_failure_is_visible_after_two_reused_chat_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = MCPPeer(monkeypatch)
    peer.fail_close = True
    runner = AgentRunner.from_config(
        mcp_agent(tmp_path), provider=StaticProvider(text_response("one"), text_response("two"))
    )
    first = threading.Event()
    second = threading.Event()
    inputs = iter(["first", "second", "/exit"])
    errors: list[str] = []

    def read(prompt: str) -> str:
        value = next(inputs)
        if value == "second":
            assert first.wait(2)
        if value == "/exit":
            assert second.wait(2)
        return value

    def output(text: str) -> None:
        if text == "one":
            first.set()
        elif text == "two":
            second.set()

    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=read,
        output_func=output,
        error_func=errors.append,
    )
    assert code == 1
    assert any("MCP" in error for error in errors)
    assert peer.events == ["open", "list", "close"]


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
    original_consume = _ChatSessionHost._consume_events

    async def delay_display(self: _ChatSessionHost) -> None:
        while not provider.requests or len(store.load_session("cli").messages) < 2:
            await asyncio.sleep(0)
        terminal_pending.set()
        assert await asyncio.to_thread(release_display.wait, 2)
        await original_consume(self)

    async def submit_and_release(
        self: _ChatSessionHost, input: str, *, mode: Literal["follow_up"] | None = None
    ) -> None:
        try:
            await original_submit(self, input, mode=mode)
        finally:
            if input == "继续":
                release_display.set()

    monkeypatch.setattr(_ChatSessionHost, "_consume_events", delay_display)
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
        live_output=None,
        output_func=outputs.append,
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
