"""before 脚本取消后的清理失败先交结算 owner，不隐式重试或执行 body。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from iris.command import (
    CommandMode,
    CommandOutcome,
    CommandOutputStats,
    CommandRequest,
    CommandScope,
    CommandStatus,
    CommandStopReceipt,
)
from iris.exceptions import IrisCommandCleanupError
from iris.harness import AgentRunner
from iris.hooks._dispatch_types import CommandHookRegistration
from iris.hooks.command import CommandHookAdapter
from iris.hooks.dispatcher import HookDispatcher
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunLimits,
    RunPhase,
    RunStopReason,
    RuntimeExecutionOptions,
    ToolCallPhase,
)
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolRegistry

from .fakes import FrozenClock, StaticProvider, build_runtime, tool_response
from .test_command_settlement import ControlledService, bind_service


class BeforeService(ControlledService):
    """真实命令适配器的受控 service，在取消收口时报告首次失败。"""

    def __init__(self) -> None:
        super().__init__()
        self.executions = 0
        self.drain_calls = 0
        self.cleanup_error: IrisCommandCleanupError | None = None
        self.release()

    async def execute(self, scope: CommandScope, request: CommandRequest) -> CommandOutcome:
        """保留已停止收据；下次显式 wait_drained 可以成功。"""
        self.executions += 1
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            self.cleanup_error = IrisCommandCleanupError(
                "before cleanup pending",
                command_outcome=CommandOutcome(
                    mode=CommandMode.NATIVE,
                    status=CommandStatus.CANCELLED,
                    exit_code=None,
                    stdout="",
                    stderr="",
                    output_stats=CommandOutputStats(0, 0, 0, 0, frozenset()),
                    duration_seconds=0,
                    cwd=str(request.cwd),
                    stop_receipt=self.receipt,
                ),
            )
            raise self.cleanup_error from error
        raise AssertionError("before command must be cancelled")

    async def wait_drained(self, receipt: CommandStopReceipt) -> None:
        """记录是否被首次失败后的同一驱动调用偷偷重试。"""
        self.drain_calls += 1
        await super().wait_drained(receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["cancel", "deadline", "timeout", "sdk"])
@pytest.mark.parametrize("sqlite", [False, True])
async def test_before_cleanup_keeps_unknown_intent_until_explicit_retry(
    tmp_path: Path, source: str, sqlite: bool
) -> None:
    body_calls: list[str] = []

    def body() -> str:
        body_calls.append("body")
        return "body"

    registry = ToolRegistry()
    registry.register_function(body, description="body")
    provider = StaticProvider(tool_response(ToolUseBlock(id="call", name="body", input={})))
    runtime = build_runtime(tmp_path, provider=provider, registry=registry)
    store = SQLiteStore(tmp_path / "lifecycle.db") if sqlite else InMemoryLifecycleStore()
    clock = FrozenClock()
    runner = AgentRunner(runtime=runtime, store=store, clock=clock)
    service = BeforeService()
    bind_service(runner, service)
    binding = runtime.environment.command_binding
    assert binding is not None
    runtime.environment = replace(
        runtime.environment,
        hook_dispatcher=HookDispatcher(
            [
                CommandHookRegistration(
                    "tool.before",
                    "before",
                    CommandHookAdapter(
                        binding=binding, workspace=tmp_path, command="controlled command"
                    ),
                )
            ]
        ),
    )
    options = AgentRunOptions(
        limits=RunLimits(deadline_at=clock.now() + timedelta(seconds=30)),
        runtime=RuntimeExecutionOptions(tool_timeout_seconds=0.1 if source == "timeout" else None),
    )
    task = asyncio.create_task(
        runner.start(AgentRunRequest(input="run", run_id="run"), options=options)
    )
    await asyncio.wait_for(service.entered.wait(), 2)
    assert store.load_tool_call("run", "call").phase is ToolCallPhase.CLAIMED
    if source == "cancel":
        runner.request_cancel("run")
    elif source == "deadline":
        clock.advance(seconds=31)
        await runner._command_deadline_due("run")
    elif source == "sdk":
        task.cancel()
    with pytest.raises(IrisCommandCleanupError, match="before cleanup pending") as caught:
        await task
    assert caught.value is service.cleanup_error
    pending = runner._command_lifecycle.pending["run"]
    assert pending.stop_reason is (None if source == "sdk" else RunStopReason.OUTCOME_UNKNOWN)
    assert pending.receipt is service.receipt
    assert pending.call_id == "call"
    assert runner.get_run("run").phase is RunPhase.ACTIVE
    assert service.drain_calls == 0 and service.executions == 1 and body_calls == []

    await runner._command_lifecycle.join(pending)
    result = await runner.recover(
        "run", expected_activation_id=runner.get_run("run").current_activation_id
    )
    assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
    assert service.drain_calls == 1 and service.executions == 1 and body_calls == []
    assert len(provider.requests) == 1
    assert store.load_tool_call("run", "call").result is None
    assert not runner._command_lifecycle.pending
    await runner.aclose()
