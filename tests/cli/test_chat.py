from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from iris.agents import AgentConfig
from iris.cli.chat import ChatOptions, run_chat_loop
from iris.cli.render import ChatRenderer
from iris.cli.trace import ChatTraceStore, TracingRuntimeProvider
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.message import LLMResponse, TextBlock
from iris.runtime import AgentRuntime
from iris.runtime.models import RuntimeStatus, RuntimeTurnResult
from iris.session import InMemorySessionStore


class FakeProvider:
    """按顺序返回响应并由 trace wrapper 记录请求。"""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses

    async def complete(self, request):  # type: ignore[no-untyped-def]
        """返回下一条响应。"""
        del request
        return self.responses.pop(0)


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="cli-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是 CLI 助手。",
    )


def _context_input() -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(
            slots=[ContextSlot(name="instructions", content="保持简洁")]
        )
    )


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        model="gpt-4o-mini",
        content=[TextBlock(text=text)],
        finish_reason="stop",
        input_tokens=2,
        output_tokens=3,
        total_tokens=5,
    )


def _runtime(trace_store: ChatTraceStore) -> AgentRuntime:
    provider = TracingRuntimeProvider(
        FakeProvider([_response("第一答复"), _response("第二答复")]),
        trace_store,
    )
    return AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=InMemorySessionStore(),
        workspace_root=Path.cwd(),
    )


def _renderer() -> tuple[ChatRenderer, Console]:
    console = Console(record=True, width=120, color_system=None)
    return ChatRenderer(console), console


def test_chat_loop_reuses_session_and_renders_trace() -> None:
    trace_store = ChatTraceStore()
    renderer, console = _renderer()
    inputs = iter(["第一轮", "第二轮", "/exit"])

    code = run_chat_loop(
        runtime=_runtime(trace_store),
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=trace_store,
        renderer=renderer,
        input_func=lambda prompt: next(inputs),
    )

    assert code == 0
    first_steps = trace_store.steps_for_turn(1)
    second_steps = trace_store.steps_for_turn(2)
    assert len(first_steps) == 1
    assert len(second_steps) == 1
    assert first_steps[0].request.messages[-1].text == "第一轮"
    assert any(message.text == "第一轮" for message in second_steps[0].request.messages)
    assert any(
        message.text == "第一答复" for message in second_steps[0].request.messages
    )

    output = console.export_text()
    assert "USER #1" in output
    assert "REQUEST 1.1" in output
    assert "RESPONSE 2.1" in output
    assert "第一答复" in output
    assert "第二答复" in output


def test_chat_loop_handles_slash_commands_without_provider_calls() -> None:
    trace_store = ChatTraceStore()
    renderer, console = _renderer()
    inputs = iter(["/help", "/trace full", "/exit"])

    code = run_chat_loop(
        runtime=_runtime(trace_store),
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=trace_store,
        renderer=renderer,
        input_func=lambda prompt: next(inputs),
    )

    assert code == 0
    assert trace_store.steps_for_turn(1) == []
    output = console.export_text()
    assert "HELP" in output
    assert "trace 已切换为 full" in output


def test_chat_loop_reuses_event_loop_across_turns() -> None:
    class LoopRecordingRuntime:
        """记录每轮调用所在的 event loop。"""

        def __init__(self) -> None:
            self.loop_ids: list[int] = []

        async def run_loop(
            self,
            user_input: str,
            *,
            options: Any = None,
        ) -> RuntimeTurnResult:
            """返回空结果并记录当前 event loop。"""
            del user_input, options
            self.loop_ids.append(id(asyncio.get_running_loop()))
            return RuntimeTurnResult(
                session_id="demo",
                run_id="run-1",
                status=RuntimeStatus.OK,
                steps=1,
            )

    trace_store = ChatTraceStore()
    renderer, _ = _renderer()
    runtime = LoopRecordingRuntime()
    inputs = iter(["第一轮", "第二轮", "/exit"])

    code = run_chat_loop(
        runtime=runtime,  # type: ignore[arg-type]
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id="demo"),
        trace_store=trace_store,
        renderer=renderer,
        input_func=lambda prompt: next(inputs),
    )

    assert code == 0
    assert len(runtime.loop_ids) == 2
    assert runtime.loop_ids[0] == runtime.loop_ids[1]


def test_chat_options_validate_trace_mode() -> None:
    with pytest.raises(ValueError, match="trace_mode"):
        ChatOptions(config_path=Path("agent.yaml"), trace_mode="verbose")  # type: ignore[arg-type]
