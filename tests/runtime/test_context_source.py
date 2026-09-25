"""动态采集在模型步骤预留之后，和运行控制及恢复共用既有边界。"""

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeRuntimeCommitPort, MutableCancellationSignal, build_runtime, start_activation

from iris.agents import AgentConfig
from iris.context import (
    ContextBuildInput,
    ContextBuildScope,
    ContextContribution,
    ContextSection,
    ContextSlot,
    ContextSnapshot,
)
from iris.exceptions import IrisConfigError
from iris.harness import AgentRunner
from iris.harness._context_access import ContextAccess
from iris.lifecycle import RuntimeExecutionOptions
from iris.message import Msg
from iris.runtime import AgentRuntime, RuntimeActivationOutcome, RuntimeCursor, RuntimeFactory
from iris.store import InMemoryLifecycleStore
from tests.runtime.test_context_projection_execution import CountingProvider


class Source:
    """记录作用域并返回宿主本步快照。"""

    def __init__(self) -> None:
        self.scopes: list[ContextBuildScope] = []

    async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
        self.scopes.append(scope)
        return ContextSnapshot((ContextContribution("active", "current-state"),))


def _runtime(source: Source, provider: CountingProvider) -> AgentRuntime:
    runtime = build_runtime(
        agent_config=AgentConfig(name="source", model="openai/test", system="stable"),
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="rules", content="stable")])
        ),
        provider=provider,
    )
    runtime.environment.context_source = source
    return runtime


@pytest.mark.asyncio
async def test_collect_after_reservation_and_recollect_on_recovery() -> None:
    class CheckedSource(Source):
        port: FakeRuntimeCommitPort

        async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
            assert "reserve_model_step" in self.port.events
            return await super().collect(scope)

    source, provider = CheckedSource(), CountingProvider()
    runtime = _runtime(source, provider)
    activation = start_activation(
        input="original", options=RuntimeExecutionOptions(include_tools=False)
    )
    for kind in ("start", "recover"):
        current = (
            activation
            if kind == "start"
            else activation.model_copy(
                update={
                    "kind": "recover",
                    "cursor": RuntimeCursor(position="before_model", step_index=0),
                }
            )
        )
        port = FakeRuntimeCommitPort(
            current, messages=[] if kind == "start" else [Msg.user("original")]
        )
        source.port = port
        result = await runtime.execute(
            current, commits=port, cancellation=MutableCancellationSignal()
        )
        assert result.outcome is RuntimeActivationOutcome.COMPLETED
        assert not any(
            message.metadata.get("context_kind") == "runtime_snapshot" for message in port.messages
        )
    assert len(source.scopes) == 2
    assert all(scope.run_input == "original" for scope in source.scopes)
    assert provider.requests[-1].messages[-1].metadata["context_kind"] == "runtime_snapshot"


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["budget", "cancel", "deadline"])
async def test_unadmitted_model_step_does_not_collect(control: str) -> None:
    source, provider = Source(), CountingProvider()
    runtime = _runtime(source, provider)
    activation = start_activation()
    port = FakeRuntimeCommitPort(activation)
    signal = MutableCancellationSignal(requested=control == "cancel")
    if control == "budget":
        port.max_model_steps = 0
    if control == "deadline":
        port.deadline = 0
    result = await runtime.execute(activation, commits=port, cancellation=signal)
    assert (
        result.outcome.value
        == {"budget": "budget_exhausted", "cancel": "cancelled", "deadline": "deadline_exceeded"}[
            control
        ]
    )
    assert not source.scopes and not provider.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError])
async def test_ordinary_collect_failure_is_context_error(error_type: type[Exception]) -> None:
    class Broken(Source):
        async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
            raise error_type("host unavailable")

    provider = CountingProvider()
    runtime = _runtime(Broken(), provider)
    activation = start_activation()
    result = await runtime.execute(
        activation,
        commits=FakeRuntimeCommitPort(activation),
        cancellation=MutableCancellationSignal(),
    )
    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.error.source == "context" and "host unavailable" in result.error.message
    assert not provider.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
async def test_collect_consumes_deadline_before_provider(
    expired: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = CountingProvider()
    activation = start_activation()
    port = FakeRuntimeCommitPort(activation, remaining_deadline_seconds=100)

    class Delayed(Source):
        async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
            port.deadline = 0 if expired else 7
            return await super().collect(scope)

    timeouts: list[float | None] = []
    original_wait = asyncio.wait_for

    async def record_wait(operation: Awaitable[Any], timeout: float | None) -> Any:
        timeouts.append(timeout)
        return await original_wait(operation, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait_for", record_wait)
    runtime = _runtime(Delayed(), provider)
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is (
        RuntimeActivationOutcome.DEADLINE_EXCEEDED
        if expired
        else RuntimeActivationOutcome.COMPLETED
    )
    assert timeouts == ([] if expired else [7])
    assert len(provider.requests) == (0 if expired else 1)


@pytest.mark.asyncio
async def test_collect_task_cancellation_propagates_without_context_error() -> None:
    entered = asyncio.Event()

    class Waiting(Source):
        async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
            entered.set()
            await asyncio.Event().wait()
            return ContextSnapshot()

    provider = CountingProvider()
    runtime = _runtime(Waiting(), provider)
    activation = start_activation()
    task = asyncio.create_task(
        runtime.execute(
            activation,
            commits=FakeRuntimeCommitPort(activation),
            cancellation=MutableCancellationSignal(),
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not provider.requests


@pytest.mark.asyncio
async def test_source_wait_is_bounded_by_reserved_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    budgets: list[asyncio.Timeout] = []
    original_timeout = asyncio.timeout

    def track_budget(delay: float | None) -> asyncio.Timeout:
        budget = original_timeout(delay)
        budgets.append(budget)
        return budget

    class Waiting(Source):
        async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
            entered.set()
            await asyncio.Event().wait()
            return ContextSnapshot()

    monkeypatch.setattr(asyncio, "timeout", track_budget)
    provider = CountingProvider()
    runtime = _runtime(Waiting(), provider)
    activation = start_activation()
    port = FakeRuntimeCommitPort(activation, remaining_deadline_seconds=100)
    task = asyncio.create_task(
        runtime.execute(activation, commits=port, cancellation=MutableCancellationSignal())
    )
    await entered.wait()
    assert budgets[0].when() is not None
    budgets[0].reschedule(asyncio.get_running_loop().time())
    result = await task
    assert result.outcome is RuntimeActivationOutcome.DEADLINE_EXCEEDED
    assert not provider.requests


def test_source_dependency_conflict_and_factory_injection(tmp_path: Path) -> None:
    source = Source()
    config = AgentConfig(
        name="source", model="openai/test", system="stable", context_policy={"enabled": False}
    )
    for factory in (AgentRunner, RuntimeFactory):
        with pytest.raises(IrisConfigError, match="context_source"):
            factory.from_config(config, provider=CountingProvider(), context_source=source)
    enabled = config.model_copy(
        update={"context_policy": config.context_policy.model_copy(update={"enabled": True})}
    )
    runtime = RuntimeFactory.from_config(
        enabled,
        provider=CountingProvider(),
        context_access=ContextAccess(InMemoryLifecycleStore()),
        context_source=source,
    )
    assert runtime.environment.context_source is source
