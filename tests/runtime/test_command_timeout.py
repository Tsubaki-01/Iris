"""命令自身期限与 runtime/run 期限分开，停止事实不依赖工具结果对象。"""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeProvider, FakeRuntimeCommitPort, MutableCancellationSignal, start_activation
from pydantic import BaseModel

from iris.command import (
    CommandBinding,
    CommandConfig,
    CommandEnvironment,
    CommandMode,
    CommandOutcome,
    CommandOutputStats,
    CommandRequest,
    CommandScope,
    CommandStatus,
    CommandStopReceipt,
)
from iris.command.service import StopOperation
from iris.exceptions import IrisCommandCleanupError, IrisToolOutcomeUnknownError
from iris.lifecycle import RuntimeExecutionOptions, ToolErrorPolicy
from iris.message import TextBlock, ToolUseBlock
from iris.runtime import RuntimeActivationOutcome
from iris.tools import (
    DefaultPermissionPolicy,
    ToolCall,
    ToolExecutionContext,
    ToolMiddleware,
    ToolNext,
    ToolRegistry,
)
from iris.tools import ToolResult as Result
from iris.tools.base import BaseTool, ToolCapability, ToolDefinition, ToolTimeoutOwner
from iris.tools.builtin.exec import ExecCommandTool
from iris.tools.builtin.python import RunPythonTool

from .test_execute import _runtime, _text_response, _tool_batch_response, _tool_response


class CommandServiceStub:
    """只控制工具结果与收尾耗时；不提供不存在的宿主隔离证据。"""

    def __init__(self, response: CommandOutcome | Exception, delay: float = 0) -> None:
        self.response = response
        self.delay = delay
        self.requests: list[CommandRequest] = []

    async def prepare(self) -> None:
        """测试服务没有外部资源。"""

    async def execute(self, scope: CommandScope, request: CommandRequest) -> CommandOutcome:
        """模拟由工具自身管理期限后的确定事实。"""
        self.requests.append(request)
        await asyncio.sleep(self.delay)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def stop(self, scope: CommandScope) -> StopOperation:
        """inner engine 不拥有 run 级环境停止。"""
        raise AssertionError("runtime must not stop the run environment")

    async def wait_drained(self, receipt: CommandStopReceipt) -> None:
        """收据结算属于 harness，不属于当前测试的 inner engine。"""
        raise AssertionError("runtime must not settle the stop operation")

    async def aclose(self) -> None:
        """测试服务没有需要关闭的资源。"""


def _outcome(status: CommandStatus, *, receipt: CommandStopReceipt | None = None) -> CommandOutcome:
    return CommandOutcome(
        CommandMode.NATIVE,
        status,
        7 if status is CommandStatus.EXITED else None,
        "partial stdout",
        "command diagnostic",
        CommandOutputStats(14, 18, 14, 18, frozenset()),
        0.03,
        ".",
        receipt,
    )


def _registry(
    service: CommandServiceStub,
    tool_type: type[ExecCommandTool] | type[RunPythonTool] = ExecCommandTool,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        tool_type(
            CommandBinding(
                CommandConfig(timeout_seconds=5),
                service,
                CommandEnvironment("Linux", CommandMode.NATIVE, "Linux", "/bin/sh"),
            )
        )
    )
    return registry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_type,tool_name,argument",
    [(ExecCommandTool, "exec_command", "command"), (RunPythonTool, "run_python", "code")],
)
async def test_tool_owned_timeout_can_finish_cleanup_and_return_to_model(
    tmp_path: Path,
    tool_type: type[ExecCommandTool] | type[RunPythonTool],
    tool_name: str,
    argument: str,
) -> None:
    service = CommandServiceStub(_outcome(CommandStatus.TIMED_OUT), delay=0.03)
    provider = FakeProvider(
        [
            _tool_response(ToolUseBlock(id="call", name=tool_name, input={argument: "x"})),
            _text_response("continued"),
        ]
    )
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=_registry(service, tool_type),
        permission_policy=DefaultPermissionPolicy(execute_mode="allow"),
    )
    activation = start_activation(options=RuntimeExecutionOptions(tool_timeout_seconds=0.01))
    commits = FakeRuntimeCommitPort(activation, remaining_deadline_seconds=0.005)
    result = await runtime.execute(
        activation, commits=commits, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert service.requests[0].timeout_seconds == 0.01
    assert commits.tool_commits[0].result.error.code == "COMMAND_TIMEOUT"
    assert len(provider.requests) == 2
    assert not runtime.environment.command_stop_slots


class ReplaceResult(ToolMiddleware):
    """模拟结果后处理创建一个新对象，停止事实仍须从当前调用槽获取。"""

    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> Result:
        """构造独立的结果副本，不持有原结果 identity。"""
        result = await call_next()
        return Result.model_validate(result.model_dump())


@pytest.mark.asyncio
async def test_last_command_stop_retains_receipt_after_result_replacement(tmp_path: Path) -> None:
    receipt = CommandStopReceipt("service", "stop")
    service = CommandServiceStub(_outcome(CommandStatus.ENVIRONMENT_INTERRUPTED, receipt=receipt))
    provider = FakeProvider(
        [_tool_response(ToolUseBlock(id="call", name="exec_command", input={"command": "x"}))]
    )
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=_registry(service),
        permission_policy=DefaultPermissionPolicy(execute_mode="allow"),
        middleware=[ReplaceResult()],
    )
    activation = start_activation(
        options=RuntimeExecutionOptions(tool_error_policy=ToolErrorPolicy.STOP)
    )
    commits = FakeRuntimeCommitPort(activation)
    result = await runtime.execute(
        activation, commits=commits, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.stop_receipt is receipt
    assert result.stop_call_id == "call"
    assert result.cursor.position == "before_model"
    assert "stop_receipt" not in result.model_dump()
    assert "service" not in str(commits.tool_commits[0].result.model_dump())


@pytest.mark.asyncio
async def test_unknown_carries_the_current_receipt_without_claim_replay(tmp_path: Path) -> None:
    receipt = CommandStopReceipt("service", "stop")
    service = CommandServiceStub(IrisToolOutcomeUnknownError("uncertain", stop_receipt=receipt))
    runtime = _runtime(
        provider=FakeProvider(
            [_tool_response(ToolUseBlock(id="call", name="exec_command", input={"command": "x"}))]
        ),
        tmp_path=tmp_path,
        registry=_registry(service),
        permission_policy=DefaultPermissionPolicy(execute_mode="allow"),
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    result = await runtime.execute(
        activation, commits=commits, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.OUTCOME_UNKNOWN
    assert result.stop_receipt is receipt
    assert result.stop_call_id == "call"
    assert not commits.tool_commits
    assert len(service.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_known_result_is_committed_before_cleanup_failure_leaves_runtime(
    tmp_path: Path,
    started: bool,
) -> None:
    cleanup = IrisCommandCleanupError(
        "stop failed",
        command_outcome=_outcome(CommandStatus.EXITED) if started else None,
        started=started,
    )
    service = CommandServiceStub(cleanup)
    provider = FakeProvider(
        [_tool_response(ToolUseBlock(id="call", name="exec_command", input={"command": "x"}))]
    )
    runtime = _runtime(
        provider=provider,
        tmp_path=tmp_path,
        registry=_registry(service),
        permission_policy=DefaultPermissionPolicy(execute_mode="allow"),
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    result = await runtime.execute(
        activation, commits=commits, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.cleanup_error is cleanup
    assert result.error is not None and result.error.code == "COMMAND_CLEANUP_FAILED"
    assert result.stop_call_id == "call"
    assert len(commits.tool_commits) == 1
    assert commits.tool_commits[0].result.error.code == (
        "COMMAND_FAILED" if started else "COMMAND_UNAVAILABLE"
    )
    assert len(provider.requests) == 1
    assert (
        runtime.environment.command_stop_slots[(activation.run_id, "call")].cleanup_error is cleanup
    )


@pytest.mark.asyncio
async def test_return_to_model_releases_old_receipt_before_later_provider_failure(
    tmp_path: Path,
) -> None:
    receipt = CommandStopReceipt("service", "old-stop")
    service = CommandServiceStub(_outcome(CommandStatus.ENVIRONMENT_INTERRUPTED, receipt=receipt))
    runtime = _runtime(
        provider=FakeProvider(
            [_tool_response(ToolUseBlock(id="call", name="exec_command", input={"command": "x"}))]
        ),
        tmp_path=tmp_path,
        registry=_registry(service),
        permission_policy=DefaultPermissionPolicy(execute_mode="allow"),
    )
    activation = start_activation()
    result = await runtime.execute(
        activation,
        commits=FakeRuntimeCommitPort(activation),
        cancellation=MutableCancellationSignal(),
    )
    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.error.source == "provider"
    assert result.stop_receipt is None
    assert not runtime.environment.command_stop_slots


@pytest.mark.asyncio
async def test_parallel_window_respects_explicit_tool_timeout_owner(tmp_path: Path) -> None:
    entered = 0
    together = asyncio.Event()

    class OwnedRead(BaseTool):
        """显式声明自有期限的只读扩展，验证并行入口消费同一契约。"""

        timeout_owner = ToolTimeoutOwner.TOOL
        definition = ToolDefinition(
            name="owned_read",
            description="read",
            input_schema={"type": "object"},
            capabilities={ToolCapability.READ},
        )

        async def arun(
            self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
        ) -> Result:
            """两个调用同时进入，然后完成超过通用期限的已知收尾。"""
            nonlocal entered
            entered += 1
            if entered == 2:
                together.set()
            await asyncio.wait_for(together.wait(), 1)
            await asyncio.sleep(0.03)
            assert context.tool_timeout_seconds == 0.01
            return Result(
                tool_use_id=context.call_id, tool_name=self.name, content=[TextBlock(text="done")]
            )

    registry = ToolRegistry()
    registry.register(OwnedRead())
    runtime = _runtime(
        provider=FakeProvider(
            [
                _tool_batch_response(
                    [
                        ToolUseBlock(id="a", name="owned_read", input={}),
                        ToolUseBlock(id="b", name="owned_read", input={}),
                    ]
                ),
                _text_response("done"),
            ]
        ),
        tmp_path=tmp_path,
        registry=registry,
    )
    activation = start_activation(options=RuntimeExecutionOptions(tool_timeout_seconds=0.01))
    commits = FakeRuntimeCommitPort(activation)
    result = await runtime.execute(
        activation, commits=commits, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert [fact.result.tool_use_id for fact in commits.tool_commits] == ["a", "b"]
    assert not runtime.environment.command_stop_slots
