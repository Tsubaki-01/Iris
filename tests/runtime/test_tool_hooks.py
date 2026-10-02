"""工具 Hooks 经过真实 Runtime 提交、STOP 策略与控制交接。"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from fakes import (
    FakeProvider,
    FakeRuntimeCommitPort,
    FakeRuntimeSteeringPort,
    MutableCancellationSignal,
    start_activation,
)

from iris.agents import AgentConfig
from iris.command import (
    CommandBinding,
    CommandConfig,
    CommandEnvironment,
    CommandMode,
    CommandOutcome,
    CommandRequest,
    CommandScope,
    CommandStatus,
    CommandStopReceipt,
)
from iris.exceptions import IrisCommandCleanupError
from iris.hooks import (
    HookEvent,
    HookRegistration,
    ToolAfterEvent,
    ToolAfterResult,
    ToolBeforeEvent,
    ToolBeforeResult,
)
from iris.hooks._dispatch_types import CommandHookRegistration
from iris.hooks.command import CommandHookAdapter
from iris.hooks.dispatcher import HookDispatcher
from iris.lifecycle import RuntimeExecutionOptions, ToolErrorPolicy
from iris.message import Msg, ToolUseBlock
from iris.runtime import AgentRuntime, RuntimeActivationOutcome, SteeringInput
from iris.runtime._assembly import RuntimeExecutionScope, assemble_runtime, resolve_runtime_boundary
from iris.tools import ToolRegistry

from .test_command_timeout import CommandServiceStub, _outcome
from .test_execute import _runtime, _text_response, _tool_batch_response


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_before_rejection_commits_and_claims_batch_steer_under_stop(
    tmp_path: Path, failure: bool
) -> None:
    effects: list[str] = []
    seen: list[HookEvent] = []

    def body() -> str:
        effects.append("body")
        return "body"

    async def before(event: HookEvent) -> ToolBeforeResult:
        seen.append(event)
        call = cast(ToolBeforeEvent, event)
        assert call.call_id in commits.claims
        if failure:
            raise RuntimeError("before failed")
        return ToolBeforeResult(deny_reason="denied by hook")

    async def after(event: HookEvent) -> None:
        pytest.fail("被拒绝的工具不应派发 after")

    registry = ToolRegistry()
    registry.register_function(body, description="确定结果")
    provider = FakeProvider(
        [
            _tool_batch_response([ToolUseBlock(id="call", name="body", input={})]),
            _text_response("done"),
        ]
    )
    dispatcher = HookDispatcher(
        [
            HookRegistration(event="tool.before", name="before", handler=before),
            HookRegistration(event="tool.after", name="after", handler=after),
        ]
    )
    original = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    runtime = AgentRuntime(replace(original.environment, hook_dispatcher=dispatcher))
    activation = start_activation(
        options=RuntimeExecutionOptions(tool_error_policy=ToolErrorPolicy.STOP)
    )
    commits = FakeRuntimeCommitPort(activation)
    steering = FakeRuntimeSteeringPort(
        [SteeringInput(submission_id="steer", message=Msg.user("next direction"))]
    )

    result = await runtime.execute(
        activation, commits=commits, cancellation=MutableCancellationSignal(), steering=steering
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert effects == []
    assert len(seen) == 1
    assert (seen[0].agent_id, seen[0].session_id, seen[0].run_id, seen[0].activation_id) == (
        runtime.environment.agent_config.name,
        activation.session_id,
        activation.run_id,
        activation.activation_id,
    )
    assert seen[0].workspace == str(tmp_path.resolve())
    committed = commits.tool_commits[0]
    assert committed.claim is not None
    assert committed.result.is_error
    assert committed.result.error is not None
    assert committed.result.error.code == ("HOOK_ERROR" if failure else "HOOK_REJECTED")
    assert len(committed.message_delta) == 2
    assert committed.message_delta[-1].text == "next direction"
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_after_feedback_commits_once_with_original_body(tmp_path: Path) -> None:
    effects: list[str] = []

    def body() -> str:
        effects.append("body")
        return "body text"

    async def first(event: HookEvent) -> ToolAfterResult:
        after = cast(ToolAfterEvent, event)
        assert after.body_status == "success" and after.result.model_content == "body text"
        return ToolAfterResult(feedback="first feedback")

    async def fail(event: HookEvent) -> None:
        raise RuntimeError("ordinary after failure")

    async def last(event: HookEvent) -> ToolAfterResult:
        assert cast(ToolAfterEvent, event).result.hook_feedback == ()
        return ToolAfterResult(feedback="last feedback")

    registry = ToolRegistry()
    registry.register_function(body, description="确定结果")
    provider = FakeProvider(
        [
            _tool_batch_response([ToolUseBlock(id="call", name="body", input={})]),
            _text_response("done"),
        ]
    )
    original = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    dispatcher = HookDispatcher(
        [
            HookRegistration(event="tool.after", name="first", handler=first),
            HookRegistration(event="tool.after", name="fail", handler=fail),
            HookRegistration(event="tool.after", name="last", handler=last),
        ]
    )
    runtime = AgentRuntime(replace(original.environment, hook_dispatcher=dispatcher))
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)

    result = await runtime.execute(
        activation, commits=commits, cancellation=MutableCancellationSignal()
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert effects == ["body"] and len(commits.tool_commits) == 1
    saved = commits.tool_commits[0].result
    assert saved.hook_feedback == ("first feedback", "last feedback")
    assert saved.model_content.startswith("body text\n[Hook feedback]")
    assert len(commits.tool_commits[0].message_delta[0].tool_results) == 1
    delivered = [
        block for message in provider.requests[1].messages for block in message.tool_results
    ]
    assert len(delivered) == 1 and delivered[0].text == saved.model_content


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["run", "deadline", "timeout", "sdk"])
async def test_after_control_commits_known_result_and_prior_feedback(
    tmp_path: Path, control: str
) -> None:
    entered = asyncio.Event()

    async def first(event: HookEvent) -> ToolAfterResult:
        return ToolAfterResult(feedback="kept feedback")

    async def blocked(event: HookEvent) -> None:
        entered.set()
        await asyncio.Event().wait()

    async def later(event: HookEvent) -> None:
        pytest.fail("控制中断后不应执行后续处理器")

    registry = ToolRegistry()
    registry.register_function(lambda: "known body", name="body", description="确定结果")
    provider = FakeProvider(
        [_tool_batch_response([ToolUseBlock(id="call", name="body", input={})])]
    )
    original = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    dispatcher = HookDispatcher(
        [
            HookRegistration(event="tool.after", name="first", handler=first),
            HookRegistration(event="tool.after", name="blocked", handler=blocked),
            HookRegistration(event="tool.after", name="later", handler=later),
        ]
    )
    runtime = AgentRuntime(replace(original.environment, hook_dispatcher=dispatcher))
    activation = start_activation(
        options=RuntimeExecutionOptions(tool_timeout_seconds=0.05 if control == "timeout" else None)
    )
    commits = FakeRuntimeCommitPort(activation)
    signal = MutableCancellationSignal()
    steering = FakeRuntimeSteeringPort()
    execution = asyncio.create_task(
        runtime.execute(activation, commits=commits, cancellation=signal, steering=steering)
    )
    await asyncio.wait_for(entered.wait(), 1)
    if control == "sdk":
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
    else:
        if control == "run":
            signal.requested = True
        elif control == "deadline":
            commits.deadline = 0
            signal.requested = True
        result = await asyncio.wait_for(execution, 1)
        assert (
            result.outcome
            is {
                "run": RuntimeActivationOutcome.CANCELLED,
                "deadline": RuntimeActivationOutcome.DEADLINE_EXCEEDED,
                "timeout": RuntimeActivationOutcome.FAILED,
            }[control]
        )
        if control == "timeout":
            assert result.error is not None and result.error.code == "TOOL_TIMEOUT"
    assert len(commits.tool_commits) == 1
    assert commits.tool_commits[0].result.hook_feedback == ("kept feedback",)
    assert commits.tool_commits[0].result.model_content.startswith("known body")
    assert not steering.events and len(provider.requests) == 1


@pytest.mark.asyncio
async def test_parallel_after_script_cleanup_stops_siblings_and_commits_known_prefix(
    tmp_path: Path,
) -> None:
    together = asyncio.Event()
    entered: set[str] = set()
    drained = asyncio.Event()
    receipt = CommandStopReceipt("service", "script-stop")
    cleanup = IrisCommandCleanupError(
        "script cleanup pending", command_outcome=_outcome(CommandStatus.EXITED, receipt=receipt)
    )

    class ScriptService(CommandServiceStub):
        """在第二个脚本仍运行时让第一个脚本报告清理失败。"""

        async def execute(self, scope: CommandScope, request: CommandRequest) -> CommandOutcome:
            assert request.stdin is not None
            call_id = json.loads(request.stdin)["call_id"]
            entered.add(call_id)
            if len(entered) == 2:
                together.set()
            await together.wait()
            if call_id == "a":
                raise cleanup
            try:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            finally:
                drained.set()

    service = ScriptService(_outcome(CommandStatus.EXITED))
    binding = CommandBinding(
        CommandConfig(),
        service,
        CommandEnvironment("Windows", CommandMode.NATIVE, "Windows", "cmd.exe"),
    )
    registry = ToolRegistry()
    registry.register_function(lambda: "known body", name="body", description="确定结果")
    provider = FakeProvider(
        [_tool_batch_response([ToolUseBlock(id=name, name="body", input={}) for name in "ab"])]
    )
    original = _runtime(provider=provider, tmp_path=tmp_path, registry=registry)
    dispatcher = HookDispatcher(
        [
            CommandHookRegistration(
                event="tool.after",
                name="script",
                handler=CommandHookAdapter(binding=binding, workspace=tmp_path, command="script"),
            )
        ]
    )
    runtime = AgentRuntime(
        replace(original.environment, command_binding=binding, hook_dispatcher=dispatcher)
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    steering = FakeRuntimeSteeringPort()

    result = await asyncio.wait_for(
        runtime.execute(
            activation, commits=commits, cancellation=MutableCancellationSignal(), steering=steering
        ),
        1,
    )

    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.cleanup_error is cleanup
    assert result.stop_receipt is receipt
    assert drained.is_set()
    assert [commit.result.tool_use_id for commit in commits.tool_commits] == ["a", "b"]
    assert runtime.environment.command_stop_slots[(activation.run_id, "a")].cleanup_error is cleanup
    assert not steering.events and len(provider.requests) == 1


@pytest.mark.asyncio
async def test_internal_assembly_injects_one_dispatcher_into_executor(tmp_path: Path) -> None:
    seen: list[str] = []

    async def deny(event: HookEvent) -> ToolBeforeResult:
        seen.append(cast(ToolBeforeEvent, event).tool_name)
        return ToolBeforeResult(deny_reason="skip command")

    registration = HookRegistration(event="tool.before", name="deny", handler=deny)
    config = AgentConfig(
        name="hook-assembly",
        model={"provider": "openai", "name": "fake"},
        system="assistant",
        permissions={"workspace": str(tmp_path), "writes": "allow", "execute": "allow"},
        tools={"builtin": ["exec.command"]},
        context_policy={"enabled": False},
        memory={"enabled": False},
    )
    provider = FakeProvider(
        [
            _tool_batch_response(
                [ToolUseBlock(id="call", name="exec_command", input={"command": "echo unused"})]
            ),
            _text_response("done"),
        ]
    )
    runtime = assemble_runtime(
        config,
        config_path=None,
        provider=provider,
        memory_service=None,
        api_key=None,
        execution_scope=RuntimeExecutionScope.ROOT,
        boundary=resolve_runtime_boundary(config),
        hooks=[registration],
    )
    activation = start_activation()
    commits = FakeRuntimeCommitPort(activation)
    try:
        await runtime.environment.aprepare()
        result = await runtime.execute(
            activation, commits=commits, cancellation=MutableCancellationSignal()
        )
    finally:
        await runtime.environment.aclose()
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert seen == ["exec_command"]
    assert commits.tool_commits[0].result.error.code == "HOOK_REJECTED"
