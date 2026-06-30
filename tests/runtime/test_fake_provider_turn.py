from __future__ import annotations

import pytest
from fakes import FakeProvider

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.exceptions import IrisAuthenticationError, IrisError, IrisSessionError
from iris.message import LLMResponse, Msg, Role, TextBlock
from iris.runtime import AgentRuntime, normalize_runtime_error
from iris.runtime.models import RuntimeOptions, RuntimeStatus
from iris.session import InMemorySessionStore


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="runtime-agent",
        model={
            "provider": "openai",
            "name": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 128,
        },
        system="你是本地助手。",
    )


def _context_input() -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(
            slots=[ContextSlot(name="instructions", content="遵守用户指令")]
        )
    )


def _assistant_response(text: str = "你好，我是 Iris。") -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id="response-1",
        model="gpt-4o-mini",
        content=[TextBlock(text=text)],
        finish_reason="stop",
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
    )


def test_normalize_runtime_error_uses_session_error_mapping() -> None:
    error = normalize_runtime_error(IrisSessionError("session 写入失败"))

    assert error.code == "SESSION_ERROR"
    assert error.source == "session"


def test_normalize_runtime_error_inherits_provider_mapping() -> None:
    error = normalize_runtime_error(IrisAuthenticationError("认证失败"))

    assert error.code == "PROVIDER_ERROR"
    assert error.source == "provider"


def test_normalize_runtime_error_keeps_plain_runtime_error_generic() -> None:
    error = normalize_runtime_error(RuntimeError("provider implementation bug"))

    assert error.code == "RUNTIME_ERROR"
    assert error.source == "runtime"


def test_normalize_runtime_error_reads_iris_error_runtime_metadata() -> None:
    class CustomRuntimeMappedError(IrisError):
        runtime_error_source = "memory"
        runtime_error_code = "MEMORY_ERROR"

    error = normalize_runtime_error(CustomRuntimeMappedError("memory failure"))

    assert error.code == "MEMORY_ERROR"
    assert error.source == "memory"


@pytest.mark.asyncio
async def test_run_turn_calls_fake_provider_once_and_saves_assistant_message() -> None:
    store = InMemorySessionStore()
    store.save_messages("default", [Msg.user("历史问题").model_dump(mode="json")])
    provider = FakeProvider([_assistant_response()])
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=store,
    )

    result = await runtime.run_turn(
        "当前问题",
        options=None,
        metadata={"trace_id": "trace-1"},
    )

    assert result.status == RuntimeStatus.OK
    assert result.session_id == "default"
    assert result.assistant_message is not None
    assert result.assistant_message.text == "你好，我是 Iris。"
    assert len(provider.requests) == 1

    request = provider.requests[0]
    assert request.model == "gpt-4o-mini"
    assert request.temperature == 0.2
    assert request.max_tokens == 128
    assert [message.role for message in request.messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.USER,
    ]
    assert "遵守用户指令" in request.messages[0].text
    assert request.messages[1].text == "历史问题"
    assert request.messages[2].text == "当前问题"

    saved_messages = store.load_messages("default")
    assert [message["role"] for message in saved_messages] == [
        "user",
        "user",
        "assistant",
    ]
    assert saved_messages[0]["content"] == "历史问题"
    assert saved_messages[1]["content"] == "当前问题"
    assert saved_messages[2]["role"] == "assistant"

    metadata = store.load_run_metadata("default")
    latest_run = metadata["latest_run"]
    assert latest_run["status"] == "ok"
    assert latest_run["provider"] == "openai"
    assert latest_run["model"] == "gpt-4o-mini"
    assert latest_run["input_tokens"] == 11
    assert latest_run["output_tokens"] == 7
    assert latest_run["total_tokens"] == 18
    assert latest_run["message_count"] == 3
    assert latest_run["trace_id"] == "trace-1"


@pytest.mark.asyncio
async def test_run_turn_uses_session_id_and_request_options() -> None:
    store = InMemorySessionStore()
    store.save_messages("session-1", [Msg.user("历史问题").model_dump(mode="json")])
    provider = FakeProvider([_assistant_response("收到。")])
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=store,
    )

    result = await runtime.run_turn(
        "当前问题",
        options=RuntimeOptions(
            session_id="session-1",
            request_options={
                "temperature": 0.1,
                "provider_options": {"reasoning_effort": "low"},
            },
        ),
    )

    assert result.status == RuntimeStatus.OK
    assert result.session_id == "session-1"
    assert provider.requests[0].temperature == 0.1
    assert provider.requests[0].provider_options == {"reasoning_effort": "low"}
    assert [message.text for message in provider.requests[0].messages[1:]] == [
        "历史问题",
        "当前问题",
    ]
    assert [message["content"] for message in store.load_messages("session-1")] == [
        "历史问题",
        "当前问题",
        [{"type": "text", "text": "收到。"}],
    ]


@pytest.mark.asyncio
async def test_run_turn_normalizes_provider_failure_without_saving_assistant() -> None:
    store = InMemorySessionStore()
    provider = FakeProvider([])
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=store,
    )

    result = await runtime.run_turn("当前问题")

    assert result.status == RuntimeStatus.ERROR
    assert result.error is not None
    assert result.error.code == "PROVIDER_ERROR"
    assert result.error.source == "provider"
    assert result.assistant_message is None
    assert len(provider.requests) == 1
    assert store.load_messages("default") == []


@pytest.mark.asyncio
async def test_run_turn_does_not_turn_untyped_provider_exception_into_provider_error() -> (
    None
):
    store = InMemorySessionStore()
    provider = BrokenProvider()
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=store,
    )

    result = await runtime.run_turn("当前问题")

    assert result.status == RuntimeStatus.ERROR
    assert result.error is not None
    assert result.error.code == "RUNTIME_ERROR"
    assert result.error.source == "runtime"
    assert result.assistant_message is None
    assert store.load_messages("default") == []


@pytest.mark.asyncio
async def test_run_turn_returns_session_error_after_save_failure() -> None:
    provider = FakeProvider([_assistant_response()])
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=FailingSaveSessionStore(),
    )

    result = await runtime.run_turn("当前问题")

    assert result.status == RuntimeStatus.ERROR
    assert result.error is not None
    assert result.error.code == "SESSION_ERROR"
    assert result.error.source == "session"
    assert result.assistant_message is not None
    assert result.assistant_message.text == "你好，我是 Iris。"
    assert len(provider.requests) == 1


class FailingSaveSessionStore(InMemorySessionStore):
    """测试保存失败的 session store。"""

    def save_messages(self, session_id: str, messages: list[dict[str, object]]) -> None:
        """模拟 provider 成功后 session 写入失败。"""
        del session_id, messages
        raise IrisSessionError("session 写入失败")


class BrokenProvider:
    """测试 provider 抛出未归一化异常时 runtime 不伪装来源。"""

    async def complete(self, request: object) -> LLMResponse:
        """模拟 provider 实现自身 bug。"""
        del request
        raise RuntimeError("provider implementation bug")
