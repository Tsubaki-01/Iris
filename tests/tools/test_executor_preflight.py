from __future__ import annotations

from pathlib import Path

import pytest

from iris.hitl import PermissionPrompt, QuestionPrompt, make_call_fingerprint
from iris.message import ToolUseBlock
from iris.tools import (
    AskQuestionTool,
    CircuitBreaker,
    DefaultPermissionPolicy,
    PermissionDecision,
    PermissionEffect,
    ToolCapability,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)


class _FixedPermissionPolicy:
    def __init__(self, effect: PermissionEffect) -> None:
        self.effect = effect

    def check(self, *args: object) -> PermissionDecision:
        del args
        reason = "" if self.effect is PermissionEffect.ALLOW else f"policy {self.effect.value}"
        return PermissionDecision(
            effect=self.effect,
            reason=reason,
            metadata={"source": "test"},
        )


class _CountingPermissionPolicy(_FixedPermissionPolicy):
    def __init__(self, effect: PermissionEffect) -> None:
        super().__init__(effect)
        self.calls = 0

    def check(self, *args: object) -> PermissionDecision:
        self.calls += 1
        return super().check(*args)


class _SpyCircuitBreaker(CircuitBreaker):
    def __init__(self) -> None:
        super().__init__()
        self.before_calls: list[str] = []
        self.after_calls: list[str] = []

    def before_call(self, tool_name: str) -> None:
        self.before_calls.append(tool_name)
        super().before_call(tool_name)

    def after_result(self, tool_name: str, result: ToolResult) -> None:
        self.after_calls.append(tool_name)
        super().after_result(tool_name, result)


def test_prepare_many_returns_human_gate_without_execution_side_effects(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    middleware_calls: list[str] = []
    breaker = _SpyCircuitBreaker()

    def write_note(content: str) -> str:
        calls.append(content)
        return "written"

    class Middleware:
        def before_call(self, *args: object) -> None:
            del args
            middleware_calls.append("before")

    registry = ToolRegistry()
    registry.register_function(
        write_note,
        description="写入笔记",
        capabilities={ToolCapability.WRITE},
    )
    executor = ToolExecutor(
        registry,
        permission_policy=DefaultPermissionPolicy(write_mode="confirm"),
        middleware=[Middleware()],
        circuit_breaker=breaker,
    )
    context = ToolExecutionContext(workspace_root=tmp_path, session_id="session_1")

    plan = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="write_note", input={"content": "hello"})],
        context,
    )

    prepared = plan.calls[0]
    assert prepared.permission is not None
    assert prepared.permission.effect is PermissionEffect.REQUIRE_HUMAN
    assert prepared.human_request is not None
    assert calls == []
    assert middleware_calls == []
    assert breaker.before_calls == []
    assert breaker.after_calls == []
    assert not (tmp_path / ".iris").exists()


def test_prepare_many_checks_policy_once_per_call() -> None:
    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="ordinary", description="ordinary")
    policy = _CountingPermissionPolicy(PermissionEffect.ALLOW)
    executor = ToolExecutor(registry, permission_policy=policy)

    executor.prepare_many(
        [
            ToolUseBlock(id="call_1", name="ordinary", input={}),
            ToolUseBlock(id="call_2", name="ordinary", input={}),
        ],
        ToolExecutionContext(workspace_root="."),
    )

    assert policy.calls == 2


@pytest.mark.asyncio
async def test_prepare_and_execute_prepared_check_policy_once_per_stage() -> None:
    calls: list[str] = []

    def write_note() -> str:
        calls.append("called")
        return "written"

    registry = ToolRegistry()
    registry.register_function(write_note, description="写入笔记")
    policy = _CountingPermissionPolicy(PermissionEffect.REQUIRE_HUMAN)
    executor = ToolExecutor(registry, permission_policy=policy)
    context = ToolExecutionContext(workspace_root=".")
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="write_note", input={})], context
    ).calls[0]
    policy.effect = PermissionEffect.ALLOW

    result = await executor.execute_prepared(
        prepared,
        context,
        approved_tool_call_id="call_1",
    )

    assert result.is_error is False
    assert calls == ["called"]
    assert policy.calls == 2


@pytest.mark.asyncio
async def test_execute_one_applies_deny_before_execution_lifecycle(tmp_path: Path) -> None:
    tool_calls: list[str] = []
    middleware_calls: list[str] = []
    breaker = _SpyCircuitBreaker()

    def write_note() -> str:
        tool_calls.append("called")
        return "written"

    class Middleware:
        def before_call(self, *args: object) -> None:
            del args
            middleware_calls.append("before")

    registry = ToolRegistry()
    registry.register_function(write_note, description="写入笔记")
    executor = ToolExecutor(
        registry,
        permission_policy=_FixedPermissionPolicy(PermissionEffect.DENY),
        middleware=[Middleware()],
        circuit_breaker=breaker,
    )

    result = await executor.execute_one(
        ToolUseBlock(id="call_1", name="write_note", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
        approved_tool_call_id="call_1",
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "PERMISSION_ERROR"
    assert tool_calls == []
    assert middleware_calls == []
    assert breaker.before_calls == []
    assert breaker.after_calls == []
    assert not (tmp_path / ".iris").exists()


@pytest.mark.asyncio
async def test_execute_prepared_rejects_approval_when_current_policy_denies() -> None:
    calls: list[str] = []

    def write_note() -> str:
        calls.append("called")
        return "written"

    registry = ToolRegistry()
    registry.register_function(write_note, description="写入笔记")
    policy = _CountingPermissionPolicy(PermissionEffect.REQUIRE_HUMAN)
    executor = ToolExecutor(registry, permission_policy=policy)
    context = ToolExecutionContext(workspace_root=".")
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="write_note", input={})], context
    ).calls[0]
    policy.effect = PermissionEffect.DENY

    result = await executor.execute_prepared(
        prepared,
        context,
        approved_tool_call_id="call_1",
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "PERMISSION_ERROR"
    assert result.error.details["effect"] == PermissionEffect.DENY.value
    assert calls == []
    assert policy.calls == 2


@pytest.mark.parametrize(
    ("effect", "expects_gate", "expects_error"),
    [
        (PermissionEffect.ALLOW, True, False),
        (PermissionEffect.DENY, False, True),
        (PermissionEffect.REQUIRE_HUMAN, False, True),
    ],
)
def test_human_tool_policy_matrix_fails_closed_without_double_gates(
    effect: PermissionEffect,
    expects_gate: bool,
    expects_error: bool,
) -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    executor = ToolExecutor(registry, permission_policy=_FixedPermissionPolicy(effect))

    prepared = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="ask_question", input={"question": "继续吗？"})],
        ToolExecutionContext(
            workspace_root=".",
            session_id="session_1",
            metadata={"run_id": "run_1"},
        ),
    ).calls[0]

    assert (prepared.human_request is not None) is expects_gate
    assert (prepared.preflight_result is not None) is expects_error
    if expects_gate:
        assert isinstance(prepared.human_request.prompt, QuestionPrompt)
    if expects_error:
        assert prepared.preflight_result.error.code == "PERMISSION_ERROR"
        assert prepared.preflight_result.error.details["effect"] == effect.value


@pytest.mark.parametrize(
    ("effect", "expects_gate", "expects_error"),
    [
        (PermissionEffect.ALLOW, False, False),
        (PermissionEffect.DENY, False, True),
        (PermissionEffect.REQUIRE_HUMAN, True, False),
    ],
)
def test_ordinary_tool_policy_matrix_applies_deny_before_gate_creation(
    effect: PermissionEffect,
    expects_gate: bool,
    expects_error: bool,
) -> None:
    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="ordinary", description="ordinary")
    executor = ToolExecutor(registry, permission_policy=_FixedPermissionPolicy(effect))

    prepared = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="ordinary", input={})],
        ToolExecutionContext(
            workspace_root=".",
            session_id="session_1",
            metadata={"run_id": "run_1"},
        ),
    ).calls[0]

    assert (prepared.human_request is not None) is expects_gate
    assert (prepared.preflight_result is not None) is expects_error
    if expects_gate:
        assert isinstance(prepared.human_request.prompt, PermissionPrompt)
    if expects_error:
        assert prepared.preflight_result.error.code == "PERMISSION_ERROR"
        assert prepared.preflight_result.error.details["effect"] == PermissionEffect.DENY.value


def test_both_gate_kinds_use_the_same_subject_fingerprint_contract() -> None:
    def ordinary(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    registry.register_function(ordinary, description="ordinary")
    context = ToolExecutionContext(
        workspace_root=".",
        session_id="session_1",
        metadata={"run_id": "run_1"},
    )
    question = (
        ToolExecutor(
            registry,
            permission_policy=_FixedPermissionPolicy(PermissionEffect.ALLOW),
        )
        .prepare_many(
            [ToolUseBlock(id="call_question", name="ask_question", input={"question": "继续吗？"})],
            context,
        )
        .calls[0]
    )
    permission = (
        ToolExecutor(
            registry,
            permission_policy=_FixedPermissionPolicy(PermissionEffect.REQUIRE_HUMAN),
        )
        .prepare_many(
            [ToolUseBlock(id="call_permission", name="ordinary", input={"value": "x"})],
            context,
        )
        .calls[0]
    )

    assert question.human_request is not None
    assert permission.human_request is not None
    for request in (question.human_request, permission.human_request):
        assert request.tool_call.fingerprint == make_call_fingerprint(
            session_id="session_1",
            run_id="run_1",
            tool_call_id=request.tool_call.tool_call_id,
            tool_name=request.tool_call.tool_name,
            arguments=request.tool_call.arguments,
            workspace_root=request.tool_call.workspace_root,
        )


@pytest.mark.asyncio
async def test_execute_prepared_allows_only_matching_human_approval() -> None:
    calls: list[str] = []

    def write_note(content: str) -> str:
        calls.append(content)
        return "written"

    registry = ToolRegistry()
    registry.register_function(
        write_note,
        description="写入笔记",
        capabilities={ToolCapability.WRITE},
    )
    executor = ToolExecutor(
        registry,
        permission_policy=DefaultPermissionPolicy(write_mode="confirm"),
    )
    context = ToolExecutionContext(workspace_root=".")
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="write_note", input={"content": "hello"})],
        context,
    ).calls[0]

    rejected = await executor.execute_prepared(
        prepared,
        context,
        approved_tool_call_id="other_call",
    )
    approved = await executor.execute_prepared(
        prepared,
        context,
        approved_tool_call_id="call_1",
    )

    assert rejected.is_error is True
    assert approved.is_error is False
    assert calls == ["hello"]


@pytest.mark.asyncio
async def test_execute_prepared_never_allows_denied_call() -> None:
    calls: list[str] = []

    def write_note() -> str:
        calls.append("called")
        return "written"

    registry = ToolRegistry()
    registry.register_function(
        write_note,
        description="写入笔记",
        capabilities={ToolCapability.WRITE},
    )
    executor = ToolExecutor(
        registry,
        permission_policy=DefaultPermissionPolicy(write_mode="deny"),
    )
    context = ToolExecutionContext(workspace_root=".")
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="write_note", input={})], context
    ).calls[0]

    result = await executor.execute_prepared(
        prepared,
        context,
        approved_tool_call_id="call_1",
    )

    assert result.is_error is True
    assert calls == []
