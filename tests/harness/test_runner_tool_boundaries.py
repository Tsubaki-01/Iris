"""真实 runtime 与两种 store 之间的工具批次边界回归。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest

from iris.harness import AgentRunner
from iris.hitl import PermissionInteractionResponse, QuestionInteractionResponse
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunEventKind,
    RunPhase,
    RunStopReason,
    ToolCallPhase,
    ToolErrorPolicy,
)
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import (
    AskQuestionTool,
    BaseTool,
    CircuitBreaker,
    DefaultPermissionPolicy,
    PermissionDecision,
    PermissionEffect,
    ToolCapability,
    ToolExecutionContext,
    ToolRegistry,
)

from .fakes import StaticProvider, build_runtime, text_response, tool_batch_response, tool_response


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> InMemoryLifecycleStore | SQLiteStore:
    """让每条回归经过两个真实 aggregate store。"""
    return (
        InMemoryLifecycleStore()
        if request.param == "memory"
        else SQLiteStore(tmp_path / "lifecycle.db")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("gate_kind", ["permission", "question"])
async def test_later_human_gate_waits_after_committed_ordinary_prefix(
    tmp_path: Path,
    store: InMemoryLifecycleStore | SQLiteStore,
    gate_kind: str,
) -> None:
    """后置人工 gate 的 waiting cursor 必须指向真实 subject，恢复不重复前缀。"""
    effects: list[str] = []

    def echo() -> str:
        effects.append("echo")
        return "echo"

    def write() -> str:
        effects.append("write")
        return "write"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    registry.register_function(write, description="写入", capabilities={ToolCapability.WRITE})
    registry.register(AskQuestionTool())
    gate = (
        ToolUseBlock(id="gate", name="write", input={})
        if gate_kind == "permission"
        else ToolUseBlock(id="gate", name="ask_question", input={"question": "继续？"})
    )
    provider = StaticProvider(
        tool_batch_response(ToolUseBlock(id="prefix", name="echo", input={}), gate),
        text_response("完成"),
    )
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=provider), store=store
    )
    waiting = await runner.start(AgentRunRequest(input="批次", run_id="batch"))

    assert waiting.run.phase is RunPhase.WAITING
    assert effects == ["echo"]
    assert waiting.pending_interaction is not None
    assert waiting.pending_interaction.tool_call_id == "gate"
    checkpoint = store.load_checkpoint("batch")
    assert checkpoint is not None
    assert checkpoint.engine_cursor["next_tool_index"] == 1
    prefix, pending = store.list_tool_calls("batch")
    assert prefix.phase is ToolCallPhase.COMMITTED
    assert pending.phase is ToolCallPhase.PREPARED

    result = await runner.resume(
        "batch",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=(
            PermissionInteractionResponse(decision="approve")
            if gate_kind == "permission"
            else QuestionInteractionResponse(answer="继续")
        ),
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert effects == (["echo", "write"] if gate_kind == "permission" else ["echo"])
    assert len(provider.requests) == 2
    assert all(record.phase is ToolCallPhase.COMMITTED for record in store.list_tool_calls("batch"))


@pytest.mark.asyncio
@pytest.mark.parametrize("effect", [PermissionEffect.ALLOW, PermissionEffect.DENY])
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_resume_refreshes_changed_permission_without_losing_subject(
    tmp_path: Path,
    store: InMemoryLifecycleStore | SQLiteStore,
    effect: PermissionEffect,
    decision: Literal["approve", "reject"],
) -> None:
    """等待期间的动态裁决变化不改变 durable subject，也不能被历史批准越过。"""

    class MutablePolicy(DefaultPermissionPolicy):
        """模拟外部权限状态变化，配置 fingerprint 保持稳定。"""

        effect: PermissionEffect = PermissionEffect.REQUIRE_HUMAN

        def check(
            self, tool: BaseTool, params: dict[str, Any], context: ToolExecutionContext
        ) -> PermissionDecision:
            """每次调用都读取当前外部裁决。"""
            return PermissionDecision(effect=self.effect, reason="当前策略")

    effects: list[str] = []

    def write() -> str:
        effects.append("write")
        return "write"

    registry = ToolRegistry()
    registry.register_function(write, description="写入", capabilities={ToolCapability.WRITE})
    policy = MutablePolicy()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            permission_policy=policy,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="write", name="write", input={})),
                text_response("完成"),
            ),
        ),
        store=store,
    )
    waiting = await runner.start(AgentRunRequest(input="写入", run_id="permission"))
    assert waiting.pending_interaction is not None
    policy.effect = effect

    result = await runner.resume(
        "permission",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=PermissionInteractionResponse(decision=decision),
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert effects == (
        ["write"] if effect is PermissionEffect.ALLOW and decision == "approve" else []
    )
    [record] = store.list_tool_calls("permission")
    assert record.phase is ToolCallPhase.COMMITTED
    assert record.result is not None
    if decision == "reject":
        assert record.result.error is not None
        assert record.result.error.code == "USER_REJECTED"
    elif effect is PermissionEffect.DENY:
        assert record.result.error is not None
        assert record.result.error.code == "PERMISSION_ERROR"


@pytest.mark.asyncio
@pytest.mark.parametrize("error_policy", [ToolErrorPolicy.RETURN_TO_MODEL, ToolErrorPolicy.STOP])
async def test_circuit_open_result_commits_without_claim(
    tmp_path: Path,
    store: InMemoryLifecycleStore | SQLiteStore,
    error_policy: ToolErrorPolicy,
) -> None:
    """真实熔断短路必须提交模型可见结果，并释放 active lane。"""
    effects: list[str] = []

    def unstable() -> str:
        effects.append("unstable")
        raise RuntimeError("失败")

    registry = ToolRegistry()
    registry.register_function(unstable, description="不稳定工具")
    runtime = build_runtime(
        tmp_path,
        registry=registry,
        provider=StaticProvider(
            tool_response(ToolUseBlock(id="first", name="unstable", input={})),
            text_response("首次结束"),
            tool_response(ToolUseBlock(id="second", name="unstable", input={})),
            text_response("熔断结束"),
        ),
    )
    runtime.environment.tool_bridge.tool_executor.circuit_breaker = CircuitBreaker(
        failure_threshold=1, cooldown_seconds=60
    )
    runner = AgentRunner(runtime=runtime, store=store)
    await runner.start(AgentRunRequest(input="首次失败", run_id="first-run"))

    result = await runner.start(
        AgentRunRequest(input="熔断", run_id="second-run"),
        options=AgentRunOptions(runtime={"tool_error_policy": error_policy}),
    )

    assert result.run.stop_reason is (
        RunStopReason.FAILED if error_policy is ToolErrorPolicy.STOP else RunStopReason.COMPLETED
    )
    assert effects == ["unstable"]
    [record] = store.list_tool_calls("second-run")
    assert record.phase is ToolCallPhase.COMMITTED
    assert record.result is not None and record.result.error is not None
    assert record.result.error.code == "CIRCUIT_OPEN"
    assert not any(
        event.kind is RunEventKind.TOOL_CALL_CLAIMED for event in store.list_events("second-run")
    )
    assert store.load_session_lane("default") is None
