"""专用 child 执行入口绕过普通 hooks，同时保留权限和最终 artifact。"""

from pathlib import Path
from types import MappingProxyType

import pytest
from fakes import MutableCancellationSignal

from iris.exceptions import IrisRunPersistenceError
from iris.hitl.models import HumanInteraction
from iris.message import TextBlock, ToolUseBlock
from iris.runtime import ToolBridge
from iris.tools import CircuitBreaker, ToolExecutor, ToolMiddleware, ToolRegistry, ToolResult
from iris.tools.permissions import DefaultPermissionPolicy, PermissionDecision, PermissionEffect
from iris.tools.subagent import (
    ChildWaiting,
    SubagentExecutionOutcome,
    SubagentInvocation,
    SubagentRoute,
    SubagentRouteTable,
    SubagentTool,
)


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

    async def before_call(self, *args: object) -> None:
        raise AssertionError("subagent entered middleware")


class ForbiddenBreaker(CircuitBreaker):
    """专用路径不能触发 circuit breaker。"""

    def before_call(self, tool_name: str) -> None:
        raise AssertionError("subagent entered circuit breaker")

    def after_result(self, tool_name: str, result: ToolResult) -> None:
        raise AssertionError("subagent recorded circuit breaker")


def _bridge(tmp_path: Path, port: RecordingPort, policy: CountingPolicy) -> ToolBridge:
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
    tool.definition.max_result_chars = 10
    registry = ToolRegistry()
    registry.register(tool)
    return ToolBridge(
        tool_view=registry.view(),
        tool_executor=ToolExecutor(
            registry,
            permission_policy=policy,
            middleware=[ForbiddenHooks()],
            circuit_breaker=ForbiddenBreaker(),
        ),
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
    text = "child text " * 10
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
    bridge = _bridge(tmp_path, port, CountingPolicy())
    kwargs = _context(tmp_path)
    prepared = bridge.prepare_subagent_continuation(
        ToolUseBlock(id="delegate", name="subagent", input={"prompt": "work"}),
        **kwargs,
    )
    assert await bridge.execute_subagent_prepared(prepared, **kwargs) == waiting
    assert not (tmp_path / ".iris").exists()
    port.error = IrisRunPersistenceError("store unavailable")
    with pytest.raises(IrisRunPersistenceError, match="store unavailable"):
        await bridge.execute_subagent_prepared(prepared, **kwargs)


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
