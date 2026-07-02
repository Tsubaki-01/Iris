from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeProvider

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.message import LLMResponse, Role, TextBlock, ToolUseBlock
from iris.runtime import AgentRuntime
from iris.runtime.models import (
    BoundedLoopOptions,
    RuntimeOptions,
    RuntimeStatus,
    ToolErrorPolicy,
)
from iris.session import InMemorySessionStore
from iris.tools import (
    DefaultPermissionPolicy,
    ToolExecutor,
    ToolRegistry,
    register_file_tools,
)


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="runtime-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
    )


def _context_input() -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(
            slots=[ContextSlot(name="instructions", content="遵守用户指令")]
        )
    )


def _assistant_text_response(text: str) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id=f"response-{text}",
        model="gpt-4o-mini",
        content=[TextBlock(text=text)],
        finish_reason="stop",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )


def _assistant_tool_response(
    *,
    tool_name: str = "echo",
    value: str = "Iris",
) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id=f"response-tool-{value}",
        model="gpt-4o-mini",
        content=[
            TextBlock(text="我需要调用工具。"),
            ToolUseBlock(id=f"call_{value}", name=tool_name, input={"value": value}),
        ],
        finish_reason="tool_calls",
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
    )


def _runtime(
    *,
    provider: FakeProvider,
    store: InMemorySessionStore,
    registry: ToolRegistry | None = None,
    tmp_path: Path,
) -> AgentRuntime:
    tool_registry = registry or ToolRegistry()
    return AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=store,
        tool_registry=tool_registry,
        tool_view=tool_registry.view(),
        tool_executor=ToolExecutor(tool_registry),
        workspace_root=tmp_path,
    )


@pytest.mark.asyncio
async def test_run_loop_stops_when_assistant_has_no_tool_calls(
    tmp_path: Path,
) -> None:
    store = InMemorySessionStore()
    provider = FakeProvider([_assistant_text_response("最终回答")])
    runtime = _runtime(provider=provider, store=store, tmp_path=tmp_path)

    result = await runtime.run_loop("当前问题")

    assert result.status == RuntimeStatus.OK
    assert result.steps == 1
    assert result.assistant_message is not None
    assert result.assistant_message.text == "最终回答"
    assert len(provider.requests) == 1
    assert [message.role for message in provider.requests[0].messages] == [
        Role.SYSTEM,
        Role.USER,
    ]
    assert provider.requests[0].messages[-1].text == "当前问题"


@pytest.mark.asyncio
async def test_run_loop_feeds_tool_result_to_next_provider_request_once(
    tmp_path: Path,
) -> None:
    def echo(value: str) -> str:
        return f"echo:{value}"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            _assistant_tool_response(value="Iris"),
            _assistant_text_response("最终回答"),
        ]
    )
    runtime = _runtime(
        provider=provider,
        store=store,
        registry=registry,
        tmp_path=tmp_path,
    )

    result = await runtime.run_loop("当前问题")

    assert result.status == RuntimeStatus.OK
    assert result.steps == 2
    assert len(provider.requests) == 2
    assert provider.requests[0].messages[-1].text == "当前问题"

    second_messages = provider.requests[1].messages
    assert [
        message.text for message in second_messages if message.role == Role.USER
    ] == [
        "当前问题",
        "",
    ]
    assert second_messages[-1].tool_results[0].content == "echo:Iris"
    assert "当前问题" not in [message.text for message in second_messages[2:]]
    assert [tool_result.model_content for tool_result in result.tool_results] == [
        "echo:Iris"
    ]


@pytest.mark.asyncio
async def test_run_loop_preserves_file_read_state_between_steps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello old\n", encoding="utf-8")
    registry = register_file_tools()
    provider = FakeProvider(
        [
            LLMResponse(
                provider="fake",
                model="gpt-4o-mini",
                content=[
                    ToolUseBlock(
                        id="read_1",
                        name="read_file",
                        input={"file_path": "notes.txt"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                provider="fake",
                model="gpt-4o-mini",
                content=[
                    ToolUseBlock(
                        id="edit_1",
                        name="edit_file",
                        input={
                            "file_path": "notes.txt",
                            "old_string": "old",
                            "new_string": "new",
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
            _assistant_text_response("完成"),
        ]
    )
    runtime = AgentRuntime(
        agent_config=AgentConfig(
            name="runtime-agent",
            model={"provider": "openai", "name": "gpt-4o-mini"},
            system="你是本地助手。",
            permissions={"workspace": ".", "writes": "allow"},
        ),
        context_input=_context_input(),
        provider=provider,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(
            registry,
            permission_policy=DefaultPermissionPolicy(allow_writes=True),
        ),
        workspace_root=tmp_path,
    )

    result = await runtime.run_loop(
        "编辑文件",
        options=RuntimeOptions(loop=BoundedLoopOptions(max_steps=3)),
    )

    assert result.status == RuntimeStatus.OK
    assert path.read_text(encoding="utf-8") == "hello new\n"


@pytest.mark.asyncio
async def test_run_loop_returns_max_steps_after_bounded_tool_bridge(
    tmp_path: Path,
) -> None:
    def echo(value: str) -> str:
        return f"echo:{value}"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    store = InMemorySessionStore()
    provider = FakeProvider([_assistant_tool_response(value="Iris")])
    runtime = _runtime(
        provider=provider,
        store=store,
        registry=registry,
        tmp_path=tmp_path,
    )

    result = await runtime.run_loop(
        "当前问题",
        options=RuntimeOptions(loop=BoundedLoopOptions(max_steps=1)),
    )

    assert result.status == RuntimeStatus.MAX_STEPS
    assert result.steps == 1
    assert result.error is not None
    assert result.error.code == "MAX_STEPS_REACHED"
    assert result.error.source == "runtime"
    assert result.metadata["max_steps"] == 1
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_run_loop_stop_policy_returns_error_after_tool_error(
    tmp_path: Path,
) -> None:
    store = InMemorySessionStore()
    provider = FakeProvider([_assistant_tool_response(tool_name="missing")])
    runtime = _runtime(provider=provider, store=store, tmp_path=tmp_path)

    result = await runtime.run_loop(
        "当前问题",
        options=RuntimeOptions(
            loop=BoundedLoopOptions(tool_error_policy=ToolErrorPolicy.STOP),
        ),
    )

    assert result.status == RuntimeStatus.ERROR
    assert result.steps == 1
    assert result.error is not None
    assert result.error.code == "TOOL_NOT_ALLOWED"
    assert result.error.source == "tool"
    assert len(provider.requests) == 1
    assert result.tool_results[0].is_error is True
