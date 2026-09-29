"""共享 Docker 停止的并发与收据验收，使用受控 driver。"""

import asyncio
from pathlib import Path

import pytest

from iris.exceptions import IrisExecutionCleanupError, IrisExecutionError
from iris.execution.config import DockerConfig
from iris.execution.docker import DockerCommandService
from iris.execution.models import CommandStatus, ExecutionScope

from .test_docker import FakeClient, FakeDockerError, driver, request, started

__all__ = ["driver"]


@pytest.mark.asyncio
async def test_queued_call_waits_for_stop_then_starts_normally(
    tmp_path: Path, driver: FakeClient
) -> None:
    containers = driver.containers
    containers.create_gate = asyncio.Event()
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    scope = ExecutionScope("r1", "s1")
    first = asyncio.create_task(service.execute(scope, request(tmp_path, "one")))
    await containers.create_entered.wait()
    queued = asyncio.create_task(
        service.execute(ExecutionScope("r2", "s2"), request(tmp_path, "two"))
    )
    await asyncio.sleep(0)
    operation = service.stop(scope)
    containers.create_gate.set()
    first_result, queued_result = await asyncio.wait_for(asyncio.gather(first, queued), 2)
    await operation.wait_drained()
    assert first_result.status is CommandStatus.CANCELLED
    assert queued_result.status is CommandStatus.EXITED
    assert queued_result.stdout == "two"
    assert queued_result.stop_receipt is None
    assert len(containers.created) == 1
    assert containers.container.stops == 1
    await service.aclose()


@pytest.mark.asyncio
async def test_start_response_loss_and_failed_cleanup_is_cleanup_error(
    tmp_path: Path, driver: FakeClient
) -> None:
    container = driver.containers.container
    container.start_response_error = True
    container.stop_failures.append(False)
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    try:
        with pytest.raises(IrisExecutionCleanupError) as caught:
            await service.execute(ExecutionScope("r", "s"), request(tmp_path, "one"))
        assert caught.value.context["started"] is False
        assert container.running
        assert not container.execs
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_own_cleanup_failure_preserves_known_result_on_typed_error(
    tmp_path: Path, driver: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    original = service._release_call
    first = True

    async def fail_release(call: object) -> None:
        nonlocal first
        if first:
            first = False
            raise IrisExecutionCleanupError("模拟本调用输出流关闭失败")
        await original(call)

    monkeypatch.setattr(service, "_release_call", fail_release)
    driver.containers.container.stop_failures.append(False)
    try:
        with pytest.raises(IrisExecutionCleanupError) as caught:
            await service.execute(ExecutionScope("r", "s"), request(tmp_path, "exit 7"))
        outcome = caught.value.command_outcome
        assert outcome is not None
        assert outcome.status is CommandStatus.EXITED
        assert outcome.exit_code == 7
        assert outcome.stdout == "exit 7"
        assert outcome.stop_receipt is None
        assert "command_outcome" not in caught.value.context
        assert driver.containers.container.stops == 1
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_shared_stop_drains_old_calls_before_restart(
    tmp_path: Path, driver: FakeClient
) -> None:
    container = driver.containers.container
    container.blocked.update({"one", "two"})
    container.stop_gate = asyncio.Event()
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    one_scope = ExecutionScope("r1", "s1")
    one = asyncio.create_task(service.execute(one_scope, request(tmp_path, "one")))
    two = asyncio.create_task(service.execute(ExecutionScope("r2", "s2"), request(tmp_path, "two")))
    await started(container, 2)
    operation = service.stop(one_scope)
    assert service.stop(one_scope) is operation
    await container.stop_entered.wait()
    following = asyncio.create_task(
        service.execute(ExecutionScope("r3", "s3"), request(tmp_path, "following"))
    )
    await asyncio.sleep(0.02)
    assert not following.done()
    assert len([item for item in container.execs if item.command is not None]) == 2
    container.stop_gate.set()
    receipt = await asyncio.wait_for(operation.wait_drained(), 2)
    first, second, third = await asyncio.gather(one, two, following)
    assert first.status is CommandStatus.CANCELLED
    assert second.status is CommandStatus.ENVIRONMENT_INTERRUPTED
    assert first.stop_receipt == second.stop_receipt == receipt
    assert third.status is CommandStatus.EXITED
    assert container.starts == 2
    assert container.stops == 1
    await service.wait_drained(receipt)
    assert container.stops == 1
    await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["create", "start", "exec-start"])
async def test_stop_recovers_inflight_control_response(
    tmp_path: Path, driver: FakeClient, stage: str
) -> None:
    containers = driver.containers
    container = containers.container
    gate = asyncio.Event()
    if stage == "create":
        containers.create_gate = gate
    elif stage == "start":
        container.start_gate = gate
    else:
        container.exec_start_gate = gate
        container.blocked.add("one")
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    scope = ExecutionScope("r", "s")
    task = asyncio.create_task(service.execute(scope, request(tmp_path, "one")))
    if stage == "create":
        await containers.create_entered.wait()
    elif stage == "start":
        await container.start_entered.wait()
    else:
        await started(container, 1)
    operation = service.stop(scope)
    waiter = asyncio.create_task(operation.wait_stopped())
    await asyncio.sleep(0.02)
    assert not waiter.done()
    gate.set()
    receipt = await asyncio.wait_for(waiter, 2)
    outcome = await asyncio.wait_for(task, 2)
    await operation.wait_drained()
    assert outcome.stop_receipt == receipt
    assert not container.running
    assert container.stops == 1
    if stage != "exec-start":
        assert not container.execs
    await service.aclose()


@pytest.mark.asyncio
async def test_repeated_cancel_joins_same_shared_stop(tmp_path: Path, driver: FakeClient) -> None:
    container = driver.containers.container
    container.blocked.add("one")
    container.stop_gate = asyncio.Event()
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    task = asyncio.create_task(service.execute(ExecutionScope("r", "s"), request(tmp_path, "one")))
    await started(container, 1)
    task.cancel()
    await container.stop_entered.wait()
    task.cancel()
    container.stop_gate.set()
    outcome = await asyncio.wait_for(task, 2)
    assert outcome.status is CommandStatus.CANCELLED
    assert outcome.stop_receipt is not None
    await service.wait_drained(outcome.stop_receipt)
    assert container.stops == 1
    await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("actually_stopped", [True, False])
async def test_stop_response_uncertain_inspects_once_and_keeps_failed_gate(
    tmp_path: Path, driver: FakeClient, actually_stopped: bool
) -> None:
    container = driver.containers.container
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    await service.execute(ExecutionScope("r", "s"), request(tmp_path, "first"))
    container.stop_failures.append(actually_stopped)
    operation = service.stop(ExecutionScope("r", "s"))
    if actually_stopped:
        await operation.wait_drained()
    else:
        with pytest.raises(IrisExecutionCleanupError):
            await operation.wait_drained()
        with pytest.raises(IrisExecutionCleanupError):
            await service.execute(ExecutionScope("r2", "s2"), request(tmp_path, "blocked"))
        assert container.starts == 1
        assert service.stop(ExecutionScope("r", "s")) is operation
        await operation.wait_drained()
    assert container.shows == 1
    await service.aclose()


@pytest.mark.asyncio
async def test_create_response_loss_recovers_only_owned_name(
    tmp_path: Path, driver: FakeClient
) -> None:
    driver.containers.create_error = True
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    with pytest.raises(IrisExecutionError):
        await service.execute(ExecutionScope("r", "s"), request(tmp_path, "one"))
    assert driver.containers.gets == [driver.containers.created[0][1]]
    assert driver.containers.container.stops == 1
    assert not driver.containers.container.execs
    await service.aclose()
    assert driver.containers.container.deleted


@pytest.mark.asyncio
async def test_close_stops_deletes_and_rejects_execution(
    tmp_path: Path, driver: FakeClient
) -> None:
    container = driver.containers.container
    container.blocked.add("one")
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    task = asyncio.create_task(service.execute(ExecutionScope("r", "s"), request(tmp_path, "one")))
    await started(container, 1)
    await asyncio.wait_for(service.aclose(), 2)
    await task
    assert container.deleted
    assert driver.closed
    with pytest.raises(IrisExecutionError):
        await service.execute(ExecutionScope("r2", "s2"), request(tmp_path, "new"))


@pytest.mark.asyncio
async def test_known_result_survives_shared_stop_failure(
    tmp_path: Path, driver: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    original = service._control_exec
    known = asyncio.Event()
    proceed = asyncio.Event()

    async def delay_delete(source: str, path: str) -> str | None:
        if "unlink" in source:
            known.set()
            await proceed.wait()
        return await original(source, path)

    monkeypatch.setattr(service, "_control_exec", delay_delete)
    task = asyncio.create_task(
        service.execute(ExecutionScope("r", "s"), request(tmp_path, "exit 7"))
    )
    await known.wait()
    driver.containers.container.stop_failures.append(False)
    operation = service.stop(ExecutionScope("other", "other"))
    with pytest.raises(IrisExecutionCleanupError):
        await operation.wait_stopped()
    proceed.set()
    outcome = await task
    assert outcome.status is CommandStatus.EXITED
    assert outcome.exit_code == 7
    assert driver.containers.container.stops == 1
    await service.aclose()


@pytest.mark.asyncio
async def test_create_unknown_without_recovered_handle_keeps_cleanup_pending(
    tmp_path: Path, driver: FakeClient
) -> None:
    driver.containers.create_error = True
    driver.containers.get_error = FakeDockerError(404, "not found yet")
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    scope = ExecutionScope("r", "s")
    with pytest.raises(IrisExecutionError):
        await service.execute(scope, request(tmp_path, "one"))
    with pytest.raises(IrisExecutionCleanupError):
        await service.stop(scope).wait_drained()
    assert not driver.containers.container.execs
    driver.containers.get_error = None
    await service.aclose()
    assert driver.containers.container.deleted


@pytest.mark.asyncio
async def test_close_failure_keeps_business_closed_and_allows_cleanup_retry(
    tmp_path: Path, driver: FakeClient
) -> None:
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    await service.execute(ExecutionScope("r", "s"), request(tmp_path, "first"))
    driver.containers.container.delete_failures = 1
    with pytest.raises(IrisExecutionCleanupError):
        await service.aclose()
    with pytest.raises(IrisExecutionError):
        await service.execute(ExecutionScope("r2", "s2"), request(tmp_path, "new"))
    assert not driver.closed
    await service.aclose()
    assert driver.containers.container.deleted
    assert driver.closed
