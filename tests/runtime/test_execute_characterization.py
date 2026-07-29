"""冻结 lifecycle 重构必须保留的现有 runtime 执行语义。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeProvider, build_runtime

import iris.runtime.runtime as runtime_module
from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.message import LLMResponse, TextBlock, ToolUseBlock
from iris.runtime.models import RuntimeOptions, RuntimeStatus
from iris.session import InMemorySessionStore
from iris.tools import ToolExecutor, ToolRegistry


def _agent_config() -> AgentConfig:
    """构造 characterization 使用的固定 Agent 配置。"""
    return AgentConfig(
        name="characterization-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
    )


def _context_input() -> ContextBuildInput:
    """构造不依赖外部资源的固定 context。"""
    return ContextBuildInput(
        system=ContextSection(slots=[ContextSlot(name="instructions", content="遵守用户指令")])
    )


def _tool_response(*calls: ToolUseBlock) -> LLMResponse:
    """构造包含固定顺序工具调用的 provider 响应。"""
    return LLMResponse(
        provider="fake",
        id="response-tools",
        model="gpt-4o-mini",
        content=[TextBlock(text="需要调用工具。"), *calls],
        finish_reason="tool_calls",
    )


def _text_response(text: str) -> LLMResponse:
    """构造终止 inner loop 的文本响应。"""
    return LLMResponse(
        provider="fake",
        id="response-final",
        model="gpt-4o-mini",
        content=[TextBlock(text=text)],
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_ordered_tool_batch_reaches_next_request_once_execution_semantics(
    tmp_path: Path,
) -> None:
    """保护同批工具顺序、单次回灌和无工具响应终止语义。"""
    executed: list[tuple[str, str]] = []

    def first_tool(value: str) -> str:
        executed.append(("first", value))
        return f"first:{value}"

    def second_tool(value: str) -> str:
        executed.append(("second", value))
        return f"second:{value}"

    registry = ToolRegistry()
    registry.register_function(first_tool, description="第一个顺序探针")
    registry.register_function(second_tool, description="第二个顺序探针")
    provider = FakeProvider(
        [
            _tool_response(
                ToolUseBlock(id="call-first", name="first_tool", input={"value": "A"}),
                ToolUseBlock(id="call-second", name="second_tool", input={"value": "B"}),
            ),
            _text_response("完成"),
        ]
    )
    runtime = build_runtime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=InMemorySessionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=tmp_path,
    )

    result = await runtime.run_loop(
        "按顺序执行",
        options=RuntimeOptions(session_id="session-ordered", run_id="run-ordered"),
    )

    assert result.status is RuntimeStatus.OK
    assert result.steps == 2
    assert executed == [("first", "A"), ("second", "B")]
    assert len(provider.requests) == 2
    assert sum(message.text == "按顺序执行" for message in provider.requests[1].messages) == 1
    feedback = [
        block for message in provider.requests[1].messages for block in message.tool_results
    ]
    assert [block.tool_use_id for block in feedback] == ["call-first", "call-second"]
    assert [block.content for block in feedback] == ["first:A", "second:B"]
    assert [tool_result.tool_use_id for tool_result in result.tool_results] == [
        "call-first",
        "call-second",
    ]


@pytest.mark.asyncio
async def test_ordinary_tool_retry_claims_before_effect_target_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """定义普通工具 effect 后崩溃不得重放的目标 contract。"""

    class SimulatedProcessCrash(BaseException):
        """模拟不会被 runtime 归一化吞掉的进程崩溃。"""

    effects: list[str] = []

    def effect_tool() -> str:
        effects.append("effect")
        return "done"

    registry = ToolRegistry()
    registry.register_function(effect_tool, description="副作用探针")
    provider = FakeProvider(
        [
            _tool_response(ToolUseBlock(id="call-effect", name="effect_tool", input={})),
            _tool_response(ToolUseBlock(id="call-effect", name="effect_tool", input={})),
            _text_response("完成"),
        ]
    )
    runtime = build_runtime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=InMemorySessionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=tmp_path,
    )
    options = RuntimeOptions(session_id="session-crash", run_id="run-crash")
    commit_tool_results = runtime_module.commit_tool_results

    def crash_after_effect(**_: object) -> object:
        raise SimulatedProcessCrash

    monkeypatch.setattr(runtime_module, "commit_tool_results", crash_after_effect)
    with pytest.raises(SimulatedProcessCrash):
        await runtime.run_loop("执行一次", options=options)
    assert effects == ["effect"]

    monkeypatch.setattr(runtime_module, "commit_tool_results", commit_tool_results)
    completed = await runtime.run_loop("执行一次", options=options)

    assert completed.status is RuntimeStatus.OK
    if effects == ["effect", "effect"]:
        pytest.xfail("现有普通工具路径没有 effect 前 durable claim，Phase 2 修复")
    assert effects == ["effect"]
