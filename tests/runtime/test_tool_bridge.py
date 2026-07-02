from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeProvider

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.exceptions import IrisSessionError, IrisToolExecutionError
from iris.message import LLMResponse, Msg, Role, TextBlock, ToolUseBlock
from iris.runtime import AgentRuntime, ToolBridge
from iris.runtime.models import RuntimeOptions, RuntimeStatus
from iris.session import InMemorySessionStore
from iris.tools import (
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
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


def _assistant_tool_response() -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id="response-1",
        model="gpt-4o-mini",
        content=[
            TextBlock(text="我需要调用工具。"),
            ToolUseBlock(id="call_1", name="echo", input={"value": "Iris"}),
        ],
        finish_reason="tool_calls",
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
    )


@pytest.mark.asyncio
async def test_tool_bridge_executes_active_calls_and_appends_events(
    tmp_path: Path,
) -> None:
    def echo(value: str) -> str:
        return f"echo:{value}"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    store = InMemorySessionStore()
    bridge = ToolBridge(
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
    )
    assistant_message = Msg.assistant(
        [
            ToolUseBlock(id="call_1", name="echo", input={"value": "Iris"}),
        ]
    )

    result = await bridge.execute_once(
        assistant_message=assistant_message,
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        agent_id="agent-1",
        workspace_root=tmp_path,
        permission_mode="default",
        session_store=store,
        metadata={"trace_id": "trace-1"},
    )

    assert [tool_result.model_content for tool_result in result.results] == [
        "echo:Iris"
    ]
    assert len(result.messages) == 1
    message = result.messages[0]
    assert message.role == Role.USER
    block = message.tool_results[0]
    assert block.tool_use_id == "call_1"
    assert block.content == "echo:Iris"
    assert block.is_error is False
    assert block.name == "echo"
    assert block.metadata["tool_name"] == "echo"

    assert result.events == store.load_tool_events("session-1")
    assert result.events == [
        {
            "type": "tool_result",
            "tool_call_id": "call_1",
            "tool_name": "echo",
            "status": "ok",
            "error": None,
            "artifact": None,
            "run_id": "run-1",
            "step_index": 0,
            "agent_id": "agent-1",
            "metadata": {"trace_id": "trace-1"},
        }
    ]


@pytest.mark.asyncio
async def test_tool_bridge_rejects_inactive_tool_without_executor(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    registry.register_function(lambda: "hidden", name="hidden", description="隐藏工具")
    executor = RecordingExecutor(registry)
    bridge = ToolBridge(
        tool_view=registry.view(deny={"hidden"}),
        tool_executor=executor,
    )
    assistant_message = Msg.assistant(
        [ToolUseBlock(id="call_1", name="hidden", input={})]
    )
    store = InMemorySessionStore()

    result = await bridge.execute_once(
        assistant_message=assistant_message,
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        agent_id="agent-1",
        workspace_root=tmp_path,
        permission_mode="default",
        session_store=store,
        metadata=None,
    )

    assert executor.calls == []
    assert result.results[0].is_error is True
    assert result.results[0].error is not None
    assert result.results[0].error.code == "TOOL_NOT_ALLOWED"
    assert result.messages[0].tool_results[0].content == (
        "Error[TOOL_NOT_ALLOWED]: 工具未暴露给当前模型: hidden"
    )
    assert store.load_tool_events("session-1")[0]["status"] == "error"
    assert store.load_tool_events("session-1")[0]["error"] == {
        "code": "TOOL_NOT_ALLOWED",
        "message": "工具未暴露给当前模型: hidden",
        "retryable": False,
        "details": {},
    }


@pytest.mark.asyncio
async def test_tool_bridge_rejects_missing_executor_results(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="echo", description="回显")
    bridge = ToolBridge(
        tool_view=registry.view(),
        tool_executor=MismatchedExecutor(registry, results=[]),
    )
    assistant_message = Msg.assistant(
        [ToolUseBlock(id="call_1", name="echo", input={})]
    )

    with pytest.raises(IrisToolExecutionError, match="工具执行结果数量不匹配"):
        await bridge.execute_once(
            assistant_message=assistant_message,
            session_id="session-1",
            run_id="run-1",
            step_index=0,
            agent_id="agent-1",
            workspace_root=tmp_path,
            permission_mode="default",
            session_store=InMemorySessionStore(),
            metadata=None,
        )


@pytest.mark.asyncio
async def test_tool_bridge_rejects_extra_executor_results(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="echo", description="回显")
    bridge = ToolBridge(
        tool_view=registry.view(),
        tool_executor=MismatchedExecutor(
            registry,
            results=[
                ToolResult(tool_use_id="call_1", tool_name="echo"),
                ToolResult(tool_use_id="call_2", tool_name="echo"),
            ],
        ),
    )
    assistant_message = Msg.assistant(
        [ToolUseBlock(id="call_1", name="echo", input={})]
    )

    with pytest.raises(IrisToolExecutionError, match="工具执行结果数量不匹配"):
        await bridge.execute_once(
            assistant_message=assistant_message,
            session_id="session-1",
            run_id="run-1",
            step_index=0,
            agent_id="agent-1",
            workspace_root=tmp_path,
            permission_mode="default",
            session_store=InMemorySessionStore(),
            metadata=None,
        )


@pytest.mark.asyncio
async def test_tool_bridge_rejects_non_json_event_metadata(tmp_path: Path) -> None:
    def echo() -> str:
        return "ok"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    bridge = ToolBridge(
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
    )
    assistant_message = Msg.assistant(
        [ToolUseBlock(id="call_1", name="echo", input={})]
    )

    with pytest.raises(
        IrisToolExecutionError,
        match="session 工具事件包含非 JSON 可序列化值",
    ) as exc_info:
        await bridge.execute_once(
            assistant_message=assistant_message,
            session_id="session-1",
            run_id="run-1",
            step_index=0,
            agent_id="agent-1",
            workspace_root=tmp_path,
            permission_mode="default",
            session_store=InMemorySessionStore(),
            metadata={"path": tmp_path},
        )
    assert exc_info.value.context["reason"] == (
        f"Object of type {type(tmp_path).__name__} is not JSON serializable"
    )


@pytest.mark.asyncio
async def test_run_turn_bridges_tool_calls_without_second_provider_call(
    tmp_path: Path,
) -> None:
    def echo(value: str) -> str:
        return f"echo:{value}"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    store = InMemorySessionStore()
    provider = FakeProvider([_assistant_tool_response()])
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=tmp_path,
    )

    result = await runtime.run_turn("当前问题", metadata={"trace_id": "trace-1"})

    assert result.status == RuntimeStatus.OK
    assert len(provider.requests) == 1
    assert provider.requests[0].tools[0]["function"]["name"] == "echo"
    assert result.assistant_message is not None
    assert result.assistant_message.has_tool_calls is True
    assert [tool_result.model_content for tool_result in result.tool_results] == [
        "echo:Iris"
    ]
    assert result.tool_result_messages[0].tool_results[0].content == "echo:Iris"
    saved_messages = store.load_messages("default")
    assert [message["role"] for message in saved_messages] == [
        "user",
        "assistant",
        "user",
    ]
    tool_result_content = saved_messages[2]["content"]
    assert isinstance(tool_result_content, list)
    assert tool_result_content[0]["type"] == "tool_result"
    assert tool_result_content[0]["tool_use_id"] == "call_1"
    assert tool_result_content[0]["content"] == "echo:Iris"
    metadata = store.load_run_metadata("default")
    assert metadata["latest_run"]["message_count"] == len(saved_messages)
    assert store.load_tool_events("default")[0]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_run_turn_rejects_tool_call_when_tools_disabled(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def echo(value: str) -> str:
        calls.append(value)
        return f"echo:{value}"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    store = InMemorySessionStore()
    provider = FakeProvider([_assistant_tool_response()])
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=tmp_path,
    )

    result = await runtime.run_turn(
        "当前问题",
        options=RuntimeOptions(include_tools=False),
    )

    assert provider.requests[0].tools == []
    assert calls == []
    assert result.tool_results[0].is_error is True
    assert result.tool_results[0].error is not None
    assert result.tool_results[0].error.code == "TOOL_NOT_ALLOWED"


@pytest.mark.asyncio
async def test_run_turn_normalizes_tool_event_append_failure(
    tmp_path: Path,
) -> None:
    def echo(value: str) -> str:
        return f"echo:{value}"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    provider = FakeProvider([_assistant_tool_response()])
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=FailingToolEventSessionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=tmp_path,
    )

    result = await runtime.run_turn("当前问题")

    assert result.status == RuntimeStatus.ERROR
    assert result.error is not None
    assert result.error.code == "SESSION_ERROR"
    assert result.error.source == "session"
    assert result.assistant_message is not None
    assert len(provider.requests) == 1
    metadata = runtime.session_store.load_run_metadata("default")
    assert metadata["latest_run"]["status"] == "error"
    assert metadata["latest_run"]["error"]["code"] == "SESSION_ERROR"


class RecordingExecutor(ToolExecutor):
    """记录是否被调用的测试执行器。"""

    def __init__(self, registry: ToolRegistry) -> None:
        super().__init__(registry)
        self.calls: list[list[ToolUseBlock]] = []

    async def execute_many(
        self,
        tool_uses: list[ToolUseBlock],
        context: ToolExecutionContext,
    ) -> list[ToolResult]:
        """记录批次并返回父类结果。"""
        self.calls.append(list(tool_uses))
        return await super().execute_many(tool_uses, context)


class MismatchedExecutor(ToolExecutor):
    """返回指定结果数量的测试执行器。"""

    def __init__(self, registry: ToolRegistry, results: list[ToolResult]) -> None:
        super().__init__(registry)
        self.results = results

    async def execute_many(
        self,
        tool_uses: list[ToolUseBlock],
        context: ToolExecutionContext,
    ) -> list[ToolResult]:
        """忽略输入并返回预设结果。"""
        del tool_uses, context
        return self.results


class FailingToolEventSessionStore(InMemorySessionStore):
    """测试工具事件追加失败的 session store。"""

    def append_tool_event(self, session_id: str, event: dict[str, object]) -> None:
        """模拟 session 工具事件写入失败。"""
        del session_id, event
        raise IrisSessionError("tool event 写入失败")
