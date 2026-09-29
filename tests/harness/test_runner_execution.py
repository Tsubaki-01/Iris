"""真实 Exec 工具与 runner 之间的停止收据和已知结果交接。"""

from pathlib import Path

import pytest

from iris.exceptions import IrisExecutionCleanupError
from iris.execution import (
    CommandOutcome,
    CommandRequest,
    CommandStatus,
    ExecutionMode,
    ExecutionScope,
    ExecutionStopReceipt,
)
from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunPhase,
    RunStopReason,
    RuntimeExecutionOptions,
    ToolCallPhase,
    ToolErrorPolicy,
)
from iris.message import LLMResponse, ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import DefaultPermissionPolicy, ToolCapability, ToolRegistry
from iris.tools.builtin.exec import ExecCommandTool

from .fakes import StaticProvider, build_runtime, text_response, tool_response
from .test_execution_settlement import ControlledService, bind_service


class ResultService(ControlledService):
    """模拟已知前台结果，统计重新执行和独立停止。"""

    def __init__(self, *, cleanup_failed: bool = False, unstarted: bool = False) -> None:
        super().__init__()
        self.cleanup_failed = cleanup_failed
        self.unstarted = unstarted
        self.executions = 0
        self.drains = 0

    async def execute(self, scope: ExecutionScope, request: CommandRequest) -> CommandOutcome:
        """以单独事实返回共享环境已停止，而不是把receipt编码到结果字典。"""
        self.executions += 1
        outcome = CommandOutcome(
            mode=ExecutionMode.NATIVE,
            status=CommandStatus.ENVIRONMENT_INTERRUPTED,
            exit_code=None,
            stdout="partial",
            stderr="",
            output_truncated=False,
            duration_seconds=0.1,
            cwd=".",
            stop_receipt=self.receipt,
        )
        if self.cleanup_failed:
            if self.unstarted:
                raise IrisExecutionCleanupError("owned resource not stopped", started=False)
            raise IrisExecutionCleanupError("owned resource not stopped", command_outcome=outcome)
        return outcome

    async def wait_drained(self, receipt: ExecutionStopReceipt) -> None:
        """此处返回已停止那一轮的排空证明，不再停新容器。"""
        self.drains += 1
        assert receipt is self.receipt


def command_runner(tmp_path: Path, service: ResultService, provider: StaticProvider) -> AgentRunner:
    """复用真实 Exec/Executor/Runtime，仅替换外部命令后端。"""
    registry = ToolRegistry()
    runtime = build_runtime(
        tmp_path,
        provider=provider,
        registry=registry,
        permission_policy=DefaultPermissionPolicy(execute_mode="allow"),
    )
    runner = AgentRunner(runtime=runtime, store=InMemoryLifecycleStore())
    bind_service(runner, service)
    registry.register(ExecCommandTool(runtime.environment.execution_binding))
    return runner


def command_response() -> LLMResponse:
    """请求一次命令。"""
    return tool_response(ToolUseBlock(id="cmd", name="exec_command", input={"command": "x"}))


@pytest.mark.asyncio
async def test_stop_policy_uses_current_receipt_without_stopping_again(tmp_path: Path) -> None:
    service = ResultService()
    runner = command_runner(tmp_path, service, StaticProvider(command_response()))
    result = await runner.start(
        AgentRunRequest(input="x", run_id="stop"),
        options=AgentRunOptions(
            runtime=RuntimeExecutionOptions(tool_error_policy=ToolErrorPolicy.STOP)
        ),
    )
    assert result.run.stop_reason is RunStopReason.FAILED
    assert service.drains == 1 and service.calls == []
    assert not runner.runtime.environment.command_stop_slots


@pytest.mark.asyncio
@pytest.mark.parametrize("unstarted", [False, True])
async def test_known_result_survives_cleanup_failure_and_retry(
    tmp_path: Path, unstarted: bool
) -> None:
    service = ResultService(cleanup_failed=True, unstarted=unstarted)
    provider = StaticProvider(command_response(), text_response("must not run"))
    runner = command_runner(tmp_path, service, provider)
    with pytest.raises(IrisExecutionCleanupError):
        await runner.start(AgentRunRequest(input="x", run_id="known"))
    assert runner.get_run("known").phase is RunPhase.ACTIVE
    record = runner.store.list_tool_calls("known")[0]
    assert record.phase is ToolCallPhase.COMMITTED
    assert runner.store.load_checkpoint("known").engine_cursor["position"] == "before_model"
    service.release()
    result = await runner.recover("known")
    assert result.run.stop_reason is RunStopReason.FAILED
    assert runner.store.list_tool_calls("known")[0] == record
    assert service.executions == 1 and len(provider.requests) == 1


@pytest.mark.asyncio
async def test_return_to_model_releases_receipt_before_later_hitl_cancel(tmp_path: Path) -> None:
    service = ResultService()
    provider = StaticProvider(
        command_response(),
        tool_response(ToolUseBlock(id="write", name="write", input={})),
    )
    runner = command_runner(tmp_path, service, provider)
    runner.runtime.environment.tool_bridge.tool_executor.registry.register_function(
        lambda: "ok",
        name="write",
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    waiting = await runner.start(AgentRunRequest(input="x", run_id="waiting"))
    assert waiting.run.phase is RunPhase.WAITING
    assert ("waiting", "cmd") not in runner.runtime.environment.command_stop_slots
    service.release()
    result = await runner.cancel("waiting")
    assert result.run.stop_reason is RunStopReason.CANCELLED
    assert len(service.calls) == 1 and service.drains == 0
