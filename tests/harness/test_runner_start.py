"""AgentRunner start-to-terminal 与环境指纹测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris.exceptions import IrisConfigError, IrisProviderError, IrisRunConflictError
from iris.harness import AgentRunner
from iris.harness._fingerprint import compute_environment_fingerprint
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunEventKind,
    RunPhase,
    RunStopReason,
    RuntimeExecutionOptions,
    ToolCallPhase,
    ToolErrorPolicy,
)
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import (
    BaseTool,
    DefaultPermissionPolicy,
    PermissionDecision,
    PermissionEffect,
    PermissionPolicy,
    ToolCapability,
    ToolExecutionContext,
    ToolRegistry,
)

from .fakes import (
    CountingAgentRuntime,
    StaticProvider,
    build_runtime,
    text_response,
    tool_response,
)


class OpaquePolicy(PermissionPolicy):
    """故意缺少 deterministic fingerprint payload 的自定义策略。"""

    def check(
        self,
        tool: BaseTool,
        params: dict[str, object],
        context: ToolExecutionContext,
    ) -> PermissionDecision:
        """允许调用；本测试只关注 fingerprint contract。"""
        del tool, params, context
        return PermissionDecision(effect=PermissionEffect.ALLOW)


class NonJsonPolicy(OpaquePolicy):
    """返回 live object 的错误 fingerprint 实现。"""

    def fingerprint_payload(self) -> dict[str, object]:
        return {"live": object()}


def test_environment_fingerprint_is_stable_for_equivalent_runtime(tmp_path: Path) -> None:
    """若摘要混入对象地址，两个等价装配会产生不同 fingerprint。"""
    first = build_runtime(tmp_path)
    second = build_runtime(tmp_path)

    assert compute_environment_fingerprint(first) == compute_environment_fingerprint(second)
    assert len(compute_environment_fingerprint(first)) == 64


@pytest.mark.parametrize(
    "dimension",
    ["agent", "context", "tool", "policy", "workspace"],
)
def test_environment_fingerprint_changes_for_resumability_drift(
    tmp_path: Path,
    dimension: str,
) -> None:
    """遗漏任一可恢复语义维度都会让 drift 未被检测。"""
    base = build_runtime(tmp_path)
    registry = ToolRegistry()
    if dimension == "tool":
        registry.register_function(lambda: "ok", name="probe", description="探针")
    changed = build_runtime(
        tmp_path / "other" if dimension == "workspace" else tmp_path,
        system_text="改变后的指令" if dimension == "context" else "遵守用户指令",
        registry=registry,
        permission_policy=(
            DefaultPermissionPolicy(write_mode="allow") if dimension == "policy" else None
        ),
        agent_name="other-agent" if dimension == "agent" else "runner-agent",
    )

    assert compute_environment_fingerprint(base) != compute_environment_fingerprint(changed)


def test_environment_fingerprint_rejects_opaque_custom_policy(tmp_path: Path) -> None:
    """按 class name 猜测策略会让内部状态漂移无法检测。"""
    runtime = build_runtime(tmp_path, permission_policy=OpaquePolicy())

    with pytest.raises(IrisConfigError, match="fingerprint"):
        compute_environment_fingerprint(runtime)


def test_environment_fingerprint_rejects_non_json_policy_payload(
    tmp_path: Path,
) -> None:
    """live object 不能通过 repr 地址混入 resumability fingerprint。"""
    runtime = build_runtime(tmp_path, permission_policy=NonJsonPolicy())

    with pytest.raises(IrisConfigError, match="JSON-safe"):
        compute_environment_fingerprint(runtime)


def test_environment_fingerprint_rejects_non_json_tool_snapshot(tmp_path: Path) -> None:
    """工具 metadata 的 live object 不能泄漏到 checkpoint fingerprint。"""
    registry = ToolRegistry()
    tool = registry.register_function(lambda: "ok", name="probe", description="探针")
    tool.definition.metadata["live"] = object()
    runtime = build_runtime(tmp_path, registry=registry)

    with pytest.raises(IrisConfigError, match="JSON-safe"):
        compute_environment_fingerprint(runtime)


@pytest.mark.asyncio
async def test_runner_start_returns_reloaded_terminal_result(tmp_path: Path) -> None:
    """若 runner 返回 transient engine fact，durable result/history/events 会不一致。"""
    store = InMemoryLifecycleStore()
    runtime = build_runtime(tmp_path)
    runner = AgentRunner(runtime=runtime, store=store)

    result = await runner.start(
        AgentRunRequest(input="你好", session_id="session-1", run_id="run-1")
    )

    assert result == store.load_result("run-1")
    assert result.run.phase is RunPhase.TERMINAL
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert result.assistant_message is not None
    assert result.assistant_message.text == "完成"
    assert [message.role.value for message in store.load_session("session-1").messages] == [
        "user",
        "assistant",
    ]
    assert [event.kind for event in runner.list_events("run-1")] == [
        RunEventKind.RUN_STARTED,
        RunEventKind.ACTIVATION_STARTED,
        RunEventKind.MODEL_STEP_RESERVED,
        RunEventKind.MODEL_STEP_COMMITTED,
        RunEventKind.RUN_TERMINAL,
    ]
    assert runner.get_run("run-1") == result.run
    assert runner.get_result("run-1") == result


@pytest.mark.asyncio
async def test_runner_calls_execute_once_while_runtime_owns_multi_step_tool_loop(
    tmp_path: Path,
) -> None:
    """若 harness 接管 tool loop，一个 activation 会重复调用 engine。"""
    effects: list[str] = []

    def echo(value: str) -> str:
        effects.append(value)
        return f"echo:{value}"

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="call-1", name="echo", input={"value": "Iris"})),
        text_response("最终完成"),
    )
    runtime = CountingAgentRuntime(build_runtime(tmp_path, registry=registry, provider=provider))
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=runtime, store=store)

    result = await runner.start(
        AgentRunRequest(input="调用工具", session_id="session-tool", run_id="run-tool")
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert runtime.execute_calls == 1
    assert len(provider.requests) == 2
    assert effects == ["Iris"]
    assert result.run.usage.model_steps_committed == 2
    assert result.run.usage.tool_calls_committed == 1
    [tool_call] = store.list_tool_calls("run-tool")
    assert tool_call.phase is ToolCallPhase.COMMITTED
    assert tool_call.result is not None
    assert tool_call.result.model_content == "echo:Iris"


@pytest.mark.asyncio
async def test_runner_maps_structured_engine_failure_to_durable_terminal(
    tmp_path: Path,
) -> None:
    """受控 provider 失败应作为结果返回，而不是逃逸异常。"""

    class FailingProvider:
        async def complete(self, request: LLMRequest) -> LLMResponse:
            del request
            raise IrisProviderError("provider 不可用", provider="fake")

    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=FailingProvider()),
        store=store,
    )

    result = await runner.start(
        AgentRunRequest(input="失败", session_id="session-fail", run_id="run-fail")
    )

    assert result == store.load_result("run-fail")
    assert result.run.stop_reason is RunStopReason.FAILED
    assert result.error is not None
    assert result.error.source == "provider"
    assert (
        sum(event.kind is RunEventKind.RUN_TERMINAL for event in store.list_events("run-fail")) == 1
    )


@pytest.mark.asyncio
async def test_runner_maps_tool_stop_policy_after_durable_error_result(
    tmp_path: Path,
) -> None:
    """STOP policy 也必须先提交工具错误事实再结算 run。"""
    provider = StaticProvider(tool_response(ToolUseBlock(id="missing-1", name="missing", input={})))
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=store,
    )

    result = await runner.start(
        AgentRunRequest(input="调用缺失工具", run_id="run-tool-stop"),
        options=AgentRunOptions(
            runtime=RuntimeExecutionOptions(tool_error_policy=ToolErrorPolicy.STOP)
        ),
    )

    assert result.run.stop_reason is RunStopReason.FAILED
    assert result.error is not None
    assert result.error.code == "TOOL_NOT_ALLOWED"
    [tool_call] = store.list_tool_calls("run-tool-stop")
    assert tool_call.phase is ToolCallPhase.COMMITTED
    assert tool_call.result is not None and tool_call.result.is_error


@pytest.mark.asyncio
async def test_runner_returns_waiting_result_and_keeps_session_lane(tmp_path: Path) -> None:
    """waiting 必须已持久化 interaction，并继续占用 session lane。"""

    def write(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="write-1", name="write", input={"value": "x"}))
    )
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=provider),
        store=store,
    )

    result = await runner.start(
        AgentRunRequest(input="写入", session_id="session-wait", run_id="run-wait")
    )

    assert result == store.load_result("run-wait")
    assert result.run.phase is RunPhase.WAITING
    assert result.pending_interaction is not None
    assert store.load_interaction(result.pending_interaction.interaction_id) is not None
    assert "run-wait" not in runner._active
    with pytest.raises(IrisRunConflictError, match="lane"):
        await runner.start(
            AgentRunRequest(
                input="冲突",
                session_id="session-wait",
                run_id="run-wait-conflict",
            )
        )
