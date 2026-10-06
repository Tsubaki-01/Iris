"""普通工具 Hooks 的真实执行顺序、结果资格与控制交接。"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from PIL import Image
from pydantic import BaseModel

from iris.command.config import CommandConfig
from iris.command.models import CommandEnvironment, CommandMode, CommandStatus, CommandStopReceipt
from iris.command.service import CommandBinding
from iris.exceptions import (
    IrisCancellationRequestedError,
    IrisCommandCleanupError,
    IrisToolExecutionError,
    IrisToolOutcomeUnknownError,
    IrisToolValidationError,
)
from iris.hooks import HookRegistration, ToolAfterEvent, ToolAfterResult, ToolBeforeResult
from iris.hooks.dispatcher import HookDispatcher
from iris.message import ImageBlock, Msg, TextBlock, ToolUseBlock
from iris.observability.service import Observability
from iris.providers.chat_completions import ChatCompletionsMapper
from iris.providers.responses import ResponsesMapper
from iris.tools import (
    BaseTool,
    CircuitBreaker,
    PermissionDecision,
    PermissionEffect,
    ToolCall,
    ToolDefinition,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolNext,
    ToolRegistry,
    ToolResult,
    ToolSearchTool,
    import_tool_image,
)
from iris.tools._execution_control import ToolExecutionControlSlot


def _result(text: str = "body") -> ToolResult:
    return ToolResult(tool_use_id="unbound", tool_name="body", content=[TextBlock(text=text)])


class _Body(BaseTool):
    definition = ToolDefinition(
        name="body", description="测试工具", input_schema={"type": "object", "properties": {}}
    )

    def __init__(self, callback: Callable[[ToolExecutionContext], Awaitable[ToolResult]]) -> None:
        self.callback = callback

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        return await self.callback(context)


class _Wrapper(ToolMiddleware):
    def __init__(self, callback: Callable[[ToolCall, ToolNext], Awaitable[ToolResult]]) -> None:
        self.callback = callback

    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
        return await self.callback(call, call_next)


def _executor(
    body: Callable[[ToolExecutionContext], Awaitable[ToolResult]],
    dispatcher: HookDispatcher,
    *,
    middleware: list[ToolMiddleware] | None = None,
    binding: CommandBinding | None = None,
    breaker: CircuitBreaker | None = None,
    observability: Observability | None = None,
) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(_Body(body))
    return ToolExecutor(
        registry,
        hook_dispatcher=dispatcher,
        middleware=middleware,
        command_binding=binding,
        circuit_breaker=breaker,
        observability=observability,
    )


def _context(tmp_path: Path, *, owned: bool = False) -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_root=tmp_path,
        agent_id="agent",
        session_id="session",
        metadata={"run_id": "run", "activation_id": "activation"},
        execution_control=ToolExecutionControlSlot() if owned else None,
    )


async def _execute(executor: ToolExecutor, context: ToolExecutionContext) -> ToolResult:
    return await executor.execute_one(ToolUseBlock(id="call", name="body", input={}), context)


@pytest.mark.asyncio
async def test_before_runs_after_refresh_and_claim_then_skips_body(tmp_path: Path) -> None:
    order: list[str] = []

    class Policy:
        def check(
            self, tool: BaseTool, params: dict[str, Any], context: ToolExecutionContext
        ) -> PermissionDecision:
            order.append("permission")
            return PermissionDecision(effect=PermissionEffect.ALLOW)

    class Guard:
        def before_effect(self, prepared: object) -> None:
            order.append("claim")

    async def body(context: ToolExecutionContext) -> ToolResult:
        pytest.fail("拒绝不进入body")

    async def before(event: object) -> ToolBeforeResult:
        order.append("before")
        return ToolBeforeResult(deny_reason="reject current")

    executor = _executor(
        body, HookDispatcher([HookRegistration(event="tool.before", name="deny", handler=before)])
    )
    executor.permission_policy = Policy()
    context = _context(tmp_path)
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call", name="body", input={})], context
    ).calls[0]
    result = await executor.execute_prepared(prepared, context, effect_guard=Guard())
    assert order == ["permission", "permission", "claim", "before"]
    assert result.is_error and result.error.code == "HOOK_REJECTED"
    assert result.tool_use_id == "call"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_image", [False, True])
async def test_after_real_feedback_overrides_middleware_and_keeps_body(
    tmp_path: Path,
    with_image: bool,
    observability: tuple[Observability, InMemorySpanExporter],
) -> None:
    observation, exporter = observability
    source = tmp_path / "image.png"
    with Image.new("RGB", (40, 20), "red") as picture:
        picture.save(source)
    context = _context(tmp_path)
    image = import_tool_image(source, context, name="工具图片")
    content = [TextBlock(text="body"), image] if with_image else [TextBlock(text="body")]

    async def body(context: ToolExecutionContext) -> ToolResult:
        return _result().model_copy(update={"content": content})

    async def wrapper(call: ToolCall, call_next: ToolNext) -> ToolResult:
        result = await call_next()
        assert result.content == content
        return result.model_copy(update={"hook_feedback": ("forged",)})

    async def after(event: ToolAfterEvent) -> ToolAfterResult:
        assert event.body_status == "success" and event.result.model_content == "body"
        assert event.result.content == content
        return ToolAfterResult(feedback="real")

    result = await _execute(
        _executor(
            body,
            HookDispatcher([HookRegistration(event="tool.after", name="after", handler=after)]),
            middleware=[_Wrapper(wrapper)],
            observability=observation,
        ),
        context,
    )
    assert result.hook_feedback == ("real",)
    assert result.to_msg().tool_results[0].text == "body\n[Hook feedback]\nreal"
    assert result.content == content
    [span] = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.UNSET
    recorded = json.loads(span.attributes["gen_ai.tool.call.result"])["parts"]
    assert recorded[-1] == {"type": "text", "content": "[Hook feedback]\nreal"}
    assert recorded[0] == {"type": "text", "content": "body"}
    assert [part["uri"] for part in recorded if part["type"] == "uri"] == (
        [image.model.path.as_uri()] if with_image else []
    )
    messages = [Msg.assistant([ToolUseBlock(id="call", name="body")]), result.to_msg()]
    chat = ChatCompletionsMapper().format_messages(messages)
    responses = ResponsesMapper().format_messages(messages)
    assert chat[1]["content"].count("[Hook feedback]\nreal") == 1
    output = responses[1]["output"]
    if with_image:
        assert [block for block in result.model_blocks if isinstance(block, ImageBlock)] == [image]
        url = "data:image/png;base64," + base64.b64encode(image.model.path.read_bytes()).decode()
        assert chat[2]["content"][-1]["image_url"]["url"] == url
        assert output == [
            {"type": "input_text", "text": "body"},
            {"type": "input_image", "image_url": url, "detail": "high"},
            {"type": "input_text", "text": "[Hook feedback]\nreal"},
        ]
    else:
        assert output == [
            {"type": "input_text", "text": "body"},
            {"type": "input_text", "text": "[Hook feedback]\nreal"},
        ]


@pytest.mark.asyncio
async def test_before_control_never_synthesizes_a_body_result(tmp_path: Path) -> None:
    original = IrisCancellationRequestedError("before cancelled")

    async def body(context: ToolExecutionContext) -> ToolResult:
        pytest.fail("控制中断不进入body")

    async def before(event: object) -> None:
        raise original

    executor = _executor(
        body, HookDispatcher([HookRegistration(event="tool.before", name="before", handler=before)])
    )
    with pytest.raises(IrisCancellationRequestedError) as caught:
        await _execute(executor, _context(tmp_path, owned=True))
    assert caught.value is original


@pytest.mark.asyncio
async def test_post_middleware_cancel_cannot_keep_synthesized_feedback(tmp_path: Path) -> None:
    entered = asyncio.Event()

    async def body(context: ToolExecutionContext) -> ToolResult:
        return _result().model_copy(update={"hook_feedback": ("forged",)})

    async def wrapper(call: ToolCall, call_next: ToolNext) -> ToolResult:
        await call_next()
        entered.set()
        await asyncio.Event().wait()
        return _result("unreachable")

    async def after(event: ToolAfterEvent) -> None:
        pytest.fail("Middleware已被取消，after不执行")

    context = _context(tmp_path, owned=True)
    executor = _executor(
        body,
        HookDispatcher([HookRegistration(event="tool.after", name="after", handler=after)]),
        middleware=[_Wrapper(wrapper)],
    )
    task = asyncio.create_task(_execute(executor, context))
    await entered.wait()
    task.cancel()
    result = await task
    assert result.model_content == "body" and result.hook_feedback == ()
    assert isinstance(context.execution_control.control.error, asyncio.CancelledError)


@pytest.mark.asyncio
@pytest.mark.parametrize("preflight", ["missing", "validation", "permission", "breaker", "short"])
async def test_after_requires_an_executed_body(tmp_path: Path, preflight: str) -> None:
    seen: list[str] = []

    async def body(context: ToolExecutionContext) -> ToolResult:
        pytest.fail("预检或短路不进入body")

    async def before(event: object) -> None:
        seen.append("before")

    async def after(event: object) -> ToolAfterResult:
        seen.append("after")
        return ToolAfterResult(feedback="unexpected")

    async def short(call: ToolCall, call_next: ToolNext) -> ToolResult:
        return _result("cached").model_copy(update={"hook_feedback": ("forged",)})

    dispatcher = HookDispatcher(
        [
            HookRegistration(event="tool.before", name="before", handler=before),
            HookRegistration(event="tool.after", name="after", handler=after),
        ]
    )
    breaker = CircuitBreaker(failure_threshold=1)
    executor = _executor(
        body,
        dispatcher,
        breaker=breaker,
        middleware=[_Wrapper(short)] if preflight == "short" else None,
    )
    call = ToolUseBlock(id="call", name="missing" if preflight == "missing" else "body", input={})
    if preflight == "breaker":
        breaker.after_result("body", _result().model_copy(update={"is_error": True}))
    elif preflight == "validation":

        def invalid(params: dict[str, Any]) -> dict[str, Any]:
            raise IrisToolValidationError("invalid")

        executor.registry.get("body").validate_input = invalid
    elif preflight == "permission":

        class Deny:
            def check(
                self, tool: BaseTool, params: dict[str, Any], context: ToolExecutionContext
            ) -> PermissionDecision:
                return PermissionDecision(effect=PermissionEffect.DENY, reason="denied")

        executor.permission_policy = Deny()
    result = await executor.execute_one(call, _context(tmp_path))
    assert seen == (["before"] if preflight == "short" else [])
    assert result.hook_feedback == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
async def test_body_error_after_status_is_independent_of_recovery(
    tmp_path: Path, recover: bool
) -> None:
    seen: list[tuple[str, bool]] = []

    async def body(context: ToolExecutionContext) -> ToolResult:
        raise IrisToolExecutionError("body failed")

    async def wrapper(call: ToolCall, call_next: ToolNext) -> ToolResult:
        try:
            return await call_next()
        except IrisToolExecutionError:
            return _result("recovered")

    async def after(event: ToolAfterEvent) -> ToolAfterResult:
        seen.append((event.body_status, event.result.is_error))
        assert event.result.tool_use_id == "call"
        return ToolAfterResult(feedback="inspection")

    breaker = CircuitBreaker(failure_threshold=2)
    executor = _executor(
        body,
        HookDispatcher([HookRegistration(event="tool.after", name="after", handler=after)]),
        middleware=[_Wrapper(wrapper)] if recover else None,
        breaker=breaker,
    )
    result = await _execute(executor, _context(tmp_path))
    assert seen == [("error", not recover)]
    assert result.hook_feedback == ("inspection",)
    assert breaker._states["body"].failure_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fact", ["cancelled", "environment_interrupted", "exited_receipt", "cleanup"]
)
async def test_real_command_control_facts_skip_after(tmp_path: Path, fact: str) -> None:
    receipt = CommandStopReceipt("service", "stop")

    async def body(context: ToolExecutionContext) -> ToolResult:
        if fact == "cleanup":
            context.command_stop_slot.record(cleanup_error=IrisCommandCleanupError("cleanup"))
        else:
            context.command_stop_slot.record(
                status=CommandStatus.EXITED if fact == "exited_receipt" else CommandStatus(fact),
                receipt=receipt,
            )
        return _result()

    async def wrapper(call: ToolCall, call_next: ToolNext) -> ToolResult:
        await call_next()
        return _result("rewritten")

    async def after(event: object) -> None:
        pytest.fail("真实命令控制不能因包装器改写而触发after")

    context = _context(tmp_path, owned=True)
    result = await _execute(
        _executor(
            body,
            HookDispatcher([HookRegistration(event="tool.after", name="after", handler=after)]),
            middleware=[_Wrapper(wrapper)],
        ),
        context,
    )
    assert result.model_content == "rewritten"
    assert context.execution_control.control is None


@pytest.mark.asyncio
async def test_timed_out_body_drains_before_after(tmp_path: Path) -> None:
    receipt = CommandStopReceipt("service", "stop")
    order: list[str] = []

    class Service:
        async def wait_drained(self, value: CommandStopReceipt) -> None:
            assert value is receipt
            order.append("drain")

    binding = CommandBinding(
        config=CommandConfig(),
        service=Service(),
        environment=CommandEnvironment("test", CommandMode.NATIVE, "test", "shell"),
    )

    async def body(context: ToolExecutionContext) -> ToolResult:
        order.append("body")
        context.command_stop_slot.record(status=CommandStatus.TIMED_OUT, receipt=receipt)
        return ToolResult(
            tool_use_id="call",
            tool_name="body",
            is_error=True,
            error=ToolErrorInfo(code="COMMAND_TIMEOUT", message="timeout"),
        )

    async def after(event: ToolAfterEvent) -> ToolAfterResult:
        order.append("after")
        assert event.body_status == "error"
        return ToolAfterResult(feedback="inspect timeout")

    context = _context(tmp_path)
    result = await _execute(
        _executor(
            body,
            HookDispatcher([HookRegistration(event="tool.after", name="after", handler=after)]),
            binding=binding,
        ),
        context,
    )
    assert order == ["body", "drain", "after"]
    assert context.command_stop_slot.receipt is None
    assert result.hook_feedback == ("inspect timeout",)


@pytest.mark.asyncio
async def test_after_unknown_without_command_resource_keeps_partial_feedback(
    tmp_path: Path,
) -> None:
    async def body(context: ToolExecutionContext) -> ToolResult:
        return _result()

    async def feedback(event: ToolAfterEvent) -> ToolAfterResult:
        return ToolAfterResult(feedback="kept")

    async def unknown(event: object) -> None:
        raise IrisToolOutcomeUnknownError("supplement failed")

    result = await _execute(
        _executor(
            body,
            HookDispatcher(
                [
                    HookRegistration(event="tool.after", name="first", handler=feedback),
                    HookRegistration(event="tool.after", name="unknown", handler=unknown),
                    HookRegistration(event="tool.after", name="not-run", handler=feedback),
                ]
            ),
        ),
        _context(tmp_path),
    )
    assert result.hook_feedback == ("kept",) and not result.is_error


@pytest.mark.asyncio
async def test_tool_search_disclosure_stays_a_real_body_fact(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_function(
        lambda: "unused", name="hidden", description="hidden search target", deferred=True
    )
    registry.register(ToolSearchTool(registry.view()))

    async def wrapper(call: ToolCall, call_next: ToolNext) -> ToolResult:
        result = await call_next()
        return result.model_copy(update={"metadata": {"context_revealed_tools": ["forged"]}})

    async def after(event: ToolAfterEvent) -> ToolAfterResult:
        return ToolAfterResult(feedback="ordinary feedback naming forged")

    executor = ToolExecutor(
        registry,
        middleware=[_Wrapper(wrapper)],
        hook_dispatcher=HookDispatcher(
            [HookRegistration(event="tool.after", name="after", handler=after)]
        ),
    )
    result = await executor.execute_one(
        ToolUseBlock(id="search", name="tool_search", input={"queries": ["hidden"]}),
        _context(tmp_path),
    )
    assert result.metadata["context_revealed_tools"] == ["hidden"]
    assert result.hook_feedback == ("ordinary feedback naming forged",)
