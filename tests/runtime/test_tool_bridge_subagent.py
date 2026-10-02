"""专用 child 执行入口绕过普通 hooks，同时保留权限和最终 artifact。"""

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest
from fakes import MutableCancellationSignal

from iris.exceptions import IrisRunPersistenceError
from iris.harness import AgentRunner
from iris.hitl.models import HumanInteraction
from iris.hooks import HookEvent, HookRegistration, ToolAfterEvent, ToolAfterResult, ToolBeforeEvent
from iris.hooks.dispatcher import HookDispatcher
from iris.lifecycle import AgentRunRequest, RunStopReason
from iris.message import TextBlock, ToolUseBlock
from iris.runtime import AgentRuntime, ToolBridge
from iris.runtime.environment import RuntimeExecutionScope
from iris.store import InMemoryLifecycleStore
from iris.tools import (
    CircuitBreaker,
    ToolCall,
    ToolExecutor,
    ToolMiddleware,
    ToolNext,
    ToolRegistry,
    ToolResult,
)
from iris.tools.permissions import DefaultPermissionPolicy, PermissionDecision, PermissionEffect
from iris.tools.subagent import (
    ChildWaiting,
    SubagentExecutionOutcome,
    SubagentInvocation,
    SubagentRoute,
    SubagentRouteTable,
    SubagentTool,
)

from ..harness.fakes import StaticProvider, build_runtime, text_response, tool_response


class RecordingPort:
    """记录唯一的 harness invocation。"""

    def __init__(self, outcome: SubagentExecutionOutcome) -> None:
        self.calls: list[SubagentInvocation] = []
        self.outcome = outcome
        self.error: IrisRunPersistenceError | None = None

    async def execute(self, invocation: SubagentInvocation) -> SubagentExecutionOutcome:
        self.calls.append(invocation)
        if self.error is not None:
            raise self.error
        return self.outcome


class CountingPolicy(DefaultPermissionPolicy):
    """可在 prepare 后收紧的 parent 策略。"""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.effect = PermissionEffect.ALLOW

    def check(self, *args: object) -> PermissionDecision:
        self.calls += 1
        return PermissionDecision(effect=self.effect, reason="parent policy")


class ForbiddenHooks(ToolMiddleware):
    """专用路径不能进入普通 middleware。"""

    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
        raise AssertionError("subagent entered middleware")


class ForbiddenBreaker(CircuitBreaker):
    """专用路径不能触发 circuit breaker。"""

    def before_call(self, tool_name: str) -> None:
        raise AssertionError("subagent entered circuit breaker")

    def after_result(self, tool_name: str, result: ToolResult) -> None:
        raise AssertionError("subagent recorded circuit breaker")


def _bridge(
    tmp_path: Path,
    port: RecordingPort,
    policy: CountingPolicy,
    *,
    hook_dispatcher: HookDispatcher | None = None,
) -> ToolBridge:
    tool = SubagentTool(
        routes=SubagentRouteTable(
            "researcher",
            MappingProxyType(
                {
                    "researcher": SubagentRoute("researcher", tmp_path / "child.yaml", "Research"),
                }
            ),
        ),
        port=port,
    )
    tool.definition.max_result_chars = 500
    registry = ToolRegistry()
    registry.register(tool)
    return ToolBridge(
        tool_view=registry.view(),
        tool_executor=ToolExecutor(
            registry,
            permission_policy=policy,
            middleware=[ForbiddenHooks()],
            circuit_breaker=ForbiddenBreaker(),
            hook_dispatcher=hook_dispatcher,
        ),
    )


def _hooks(seen: list[tuple[str, str, str]], owner: str) -> HookDispatcher:
    """记录真实事件身份，用不同反馈区分父与 child 的处理器。"""

    async def record(event: HookEvent) -> ToolAfterResult | None:
        assert isinstance(event, (ToolBeforeEvent, ToolAfterEvent))
        seen.append((event.event, event.agent_id, event.call_id))
        if isinstance(event, ToolAfterEvent):
            return ToolAfterResult(feedback=f"{owner} feedback")
        return None

    return HookDispatcher(
        [
            HookRegistration(event="tool.before", name=f"{owner}:before", handler=record),
            HookRegistration(event="tool.after", name=f"{owner}:after", handler=record),
        ]
    )


def _context(tmp_path: Path) -> dict[str, object]:
    return dict(
        session_id="parent-session",
        run_id="parent-run",
        agent_id="parent",
        workspace_root=tmp_path,
        permission_mode="default",
        metadata=None,
        cancellation=MutableCancellationSignal(),
    )


@pytest.mark.asyncio
async def test_linked_special_path_skips_outer_policy_and_preserves_artifact_identity(
    tmp_path: Path,
) -> None:
    text = "child text " * 100
    port = RecordingPort(ToolResult(tool_use_id="", tool_name="", content=[TextBlock(text=text)]))
    policy = CountingPolicy()
    policy.effect = PermissionEffect.DENY
    bridge = _bridge(tmp_path, port, policy)
    kwargs = _context(tmp_path)
    prepared = bridge.prepare_subagent_continuation(
        ToolUseBlock(id="delegate", name="subagent", input={"prompt": "work"}),
        **kwargs,
    )
    result = await bridge.execute_subagent_prepared(prepared, **kwargs, linked_continuation=True)
    assert policy.calls == 0
    assert len(port.calls) == 1
    invocation = port.calls[0]
    assert invocation.parent_call.parent_run_id == "parent-run"
    assert invocation.parent_call.parent_tool_call_id == "delegate"
    assert not invocation.cancellation.requested
    assert result.tool_use_id == "delegate"
    assert result.tool_name == "subagent"
    assert result.artifact.path.read_text(encoding="utf-8") == text
    assert result.artifact.path.is_relative_to(tmp_path / ".iris" / "tool-results")


@pytest.mark.asyncio
async def test_fresh_special_refresh_blocks_dispatch_when_policy_tightens(tmp_path: Path) -> None:
    port = RecordingPort(ToolResult(tool_use_id="", tool_name="subagent"))
    policy = CountingPolicy()
    bridge = _bridge(tmp_path, port, policy)
    kwargs = _context(tmp_path)
    prepared = bridge.prepare_subagent_continuation(
        ToolUseBlock(id="delegate", name="subagent", input={"prompt": "work"}),
        **kwargs,
    )
    policy.effect = PermissionEffect.DENY
    result = await bridge.execute_subagent_prepared(prepared, **kwargs)
    assert result.error.code == "PERMISSION_ERROR"
    assert policy.calls == 1
    assert port.calls == []


@pytest.mark.asyncio
async def test_special_waiting_is_control_result_and_persistence_errors_propagate(
    tmp_path: Path,
) -> None:
    interaction = HumanInteraction.model_validate(
        {
            "session_id": "child-session",
            "run_id": "child",
            "step_index": 0,
            "tool_call_id": "question",
            "request": {
                "tool_call": {
                    "tool_call_id": "question",
                    "tool_name": "ask_question",
                    "arguments": {},
                    "workspace_root": str(tmp_path),
                    "fingerprint": "0" * 64,
                },
                "prompt": {"kind": "question", "question": "Which file?"},
            },
        }
    )
    waiting = ChildWaiting("child", interaction, None, None)
    port = RecordingPort(waiting)
    parent_events: list[tuple[str, str, str]] = []
    bridge = _bridge(
        tmp_path, port, CountingPolicy(), hook_dispatcher=_hooks(parent_events, "parent")
    )
    kwargs = _context(tmp_path)
    prepared = bridge.prepare_subagent_continuation(
        ToolUseBlock(id="delegate", name="subagent", input={"prompt": "work"}),
        **kwargs,
    )
    assert await bridge.execute_subagent_prepared(prepared, **kwargs) == waiting
    assert parent_events == []
    assert not (tmp_path / ".iris").exists()
    port.error = IrisRunPersistenceError("store unavailable")
    with pytest.raises(IrisRunPersistenceError, match="store unavailable"):
        await bridge.execute_subagent_prepared(prepared, **kwargs)
    assert parent_events == []


@pytest.mark.asyncio
async def test_parent_delegation_skips_hooks_and_child_tool_uses_own_dispatcher(
    tmp_path: Path,
) -> None:
    """父专用入口不派发 Hook；它驱动的真实 child Runner 独立提交自己的反馈。"""
    parent_events: list[tuple[str, str, str]] = []
    child_events: list[tuple[str, str, str]] = []
    bodies: list[str] = []

    def inspect_child() -> str:
        """普通 child 工具必须经过真实执行器。"""
        bodies.append("child body")
        return "child body"

    registry = ToolRegistry()
    registry.register_function(inspect_child, description="查看 child 内容")
    child_provider = StaticProvider(
        tool_response(ToolUseBlock(id="child-call", name="inspect_child", input={})),
        text_response("child complete"),
    )
    base = build_runtime(tmp_path, registry=registry, provider=child_provider, agent_name="child")
    child_runner = AgentRunner(
        runtime=AgentRuntime(
            replace(
                base.environment,
                execution_scope=RuntimeExecutionScope.CHILD,
                hook_dispatcher=_hooks(child_events, "child"),
            )
        ),
        store=InMemoryLifecycleStore(),
    )

    class ChildPort(RecordingPort):
        """只连接专用委派入口与真实 child Runner，不替代 child 工具执行。"""

        async def execute(self, invocation: SubagentInvocation) -> SubagentExecutionOutcome:
            self.calls.append(invocation)
            result = await child_runner.start(
                AgentRunRequest(
                    input=invocation.call.prompt, run_id="child-run", session_id="child-session"
                )
            )
            assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
            return ToolResult(
                tool_use_id="",
                tool_name="subagent",
                content=[TextBlock(text=result.assistant_message.text)],
            )

    port = ChildPort(ToolResult(tool_use_id="", tool_name="subagent"))
    bridge = _bridge(
        tmp_path, port, CountingPolicy(), hook_dispatcher=_hooks(parent_events, "parent")
    )
    kwargs = _context(tmp_path)
    prepared = bridge.prepare_subagent_continuation(
        ToolUseBlock(id="delegate", name="subagent", input={"prompt": "child task"}), **kwargs
    )
    try:
        result = await bridge.execute_subagent_prepared(prepared, **kwargs)
    finally:
        await child_runner.aclose()
    assert isinstance(result, ToolResult)
    assert result.model_content == "child complete" and result.hook_feedback == ()
    assert parent_events == []
    assert child_events == [
        ("tool.before", "child", "child-call"),
        ("tool.after", "child", "child-call"),
    ]
    assert bodies == ["child body"] and len(port.calls) == 1
    record = child_runner.store.load_tool_call("child-run", "child-call")
    assert record is not None and record.result is not None
    assert record.result.hook_feedback == ("child feedback",)
    delivered = [
        block for message in child_provider.requests[1].messages for block in message.tool_results
    ]
    assert (
        len(delivered) == 1
        and delivered[0].content == "child body\n[Hook feedback]\nchild feedback"
    )


def test_linked_prepare_still_rejects_raw_invalid_input(tmp_path: Path) -> None:
    port = RecordingPort(ToolResult(tool_use_id="", tool_name="subagent"))
    policy = CountingPolicy()
    bridge = _bridge(tmp_path, port, policy)
    prepared = bridge.prepare_subagent_continuation(
        ToolUseBlock(id="delegate", name="subagent", input={"prompt": " "}),
        **_context(tmp_path),
    )
    assert prepared.preflight_result.error.code == "VALIDATION_ERROR"
    assert policy.calls == 0
