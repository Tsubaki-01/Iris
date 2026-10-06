"""SubagentTool 的模型输入与专用调用契约。"""

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from iris.exceptions import IrisRunPersistenceError, IrisToolExecutionError, IrisToolValidationError
from iris.hitl.models import HumanInteraction
from iris.message import TextBlock, ToolUseBlock
from iris.observability.service import Observability
from iris.tools import (
    CircuitBreaker,
    ToolCall,
    ToolExecutor,
    ToolMiddleware,
    ToolNext,
    ToolRegistry,
)
from iris.tools.base import ToolCapability, ToolErrorInfo, ToolExecutionContext, ToolResult
from iris.tools.subagent import (
    ChildWaiting,
    SubagentExecutionOutcome,
    SubagentInvocation,
    SubagentParentCall,
    SubagentRoute,
    SubagentRouteTable,
    SubagentTool,
)


class RecordingPort:
    """记录实际工具投影到 harness 的调用。"""

    def __init__(self) -> None:
        self.calls: list[SubagentInvocation] = []
        self.outcome: SubagentExecutionOutcome = ToolResult(tool_use_id="", tool_name="subagent")

    async def execute(self, invocation: SubagentInvocation) -> SubagentExecutionOutcome:
        self.calls.append(invocation)
        return self.outcome


@pytest.fixture
def configured_tool(tmp_path: Path) -> tuple[SubagentTool, RecordingPort]:
    routes = SubagentRouteTable(
        default="researcher",
        routes=MappingProxyType(
            {
                key: SubagentRoute(key, tmp_path / f"{key}.yaml", description)
                for key, description in [
                    ("researcher", "Search notes"),
                    ("reviewer", "Review code"),
                ]
            }
        ),
    )
    port = RecordingPort()
    return SubagentTool(routes=routes, port=port), port


def test_subagent_model_schema(configured_tool: tuple[SubagentTool, RecordingPort]) -> None:
    tool, _ = configured_tool
    assert tool.name == "subagent"
    assert tool.definition.group == "agent"
    assert tool.definition.capabilities == {ToolCapability.AGENT}
    assert not tool.is_read_only({})
    assert tool.input_schema == {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "agent": {"type": "string", "enum": ["researcher", "reviewer"]},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }
    for text in ["researcher", "reviewer", "Search notes", "Review code", "default"]:
        assert text in tool.definition.description


@pytest.mark.parametrize(
    "raw",
    [
        {"prompt": " "},
        {"prompt": "x", "extra": 1},
        *[
            {"prompt": "x", "agent": agent}
            for agent in ["Researcher", " researcher", "researcher ", "x"]
        ],
    ],
)
def test_subagent_rejects_invalid_raw_input(
    configured_tool: tuple[SubagentTool, RecordingPort],
    raw: dict[str, Any],
) -> None:
    tool, port = configured_tool
    with pytest.raises(IrisToolValidationError):
        tool.validate_input(raw)
    assert port.calls == []


class RequestedCancellation:
    """供工具调用观察的协作取消信号。"""

    requested = True

    def raise_if_requested(self) -> None:
        raise IrisToolExecutionError("cancelled")


def _child_interaction(tmp_path: Path) -> HumanInteraction:
    return HumanInteraction.model_validate(
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


@pytest.mark.asyncio
async def test_waiting_outcome_and_cancellation_flow_through_dedicated_port(
    configured_tool: tuple[SubagentTool, RecordingPort],
    tmp_path: Path,
) -> None:
    tool, port = configured_tool
    interaction = _child_interaction(tmp_path)
    port.outcome = ChildWaiting("child", interaction, None, None)
    outcome = await tool.execute_subagent(
        tool.validate_input({"prompt": "investigate"}),
        ToolExecutionContext(workspace_root=tmp_path, cancellation=RequestedCancellation()),
        parent_call=SubagentParentCall("parent", "call"),
    )
    assert outcome == port.outcome
    assert len(port.calls) == 1
    assert port.calls[0].call.route.selector == "researcher"
    assert port.calls[0].cancellation.requested


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["waiting", "completed", "failed"])
async def test_dedicated_executor_observes_final_outcome_without_ordinary_pipeline(
    configured_tool: tuple[SubagentTool, RecordingPort],
    tmp_path: Path,
    observability: tuple[Observability, InMemorySpanExporter],
    status: str,
) -> None:
    observation, exporter = observability
    tool, port = configured_tool
    if status == "waiting":
        port.outcome = ChildWaiting("child", _child_interaction(tmp_path), None, None)
    else:
        port.outcome = ToolResult(
            tool_use_id="",
            tool_name="subagent",
            content=[TextBlock(text="child result")],
            is_error=status == "failed",
            error=ToolErrorInfo(code="CHILD_FAILED", message="child failed")
            if status == "failed"
            else None,
        )

    class MustNotRun(ToolMiddleware):
        async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
            pytest.fail("subagent 专用入口不能进入普通 middleware")

    breaker = CircuitBreaker(failure_threshold=1)
    breaker.after_result(
        "subagent", ToolResult(tool_use_id="old", tool_name="subagent", is_error=True)
    )
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(
        registry,
        middleware=[MustNotRun()],
        circuit_breaker=breaker,
        observability=observation,
    )
    context = ToolExecutionContext(workspace_root=tmp_path)
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call", name="subagent", input={"prompt": "  investigate  "})], context
    ).calls[0]
    with observation.scope("parent") as parent:
        outcome = await executor.execute_subagent_prepared(
            prepared, context, parent_call=SubagentParentCall("parent-run", "call")
        )
        assert trace.get_current_span() is parent
    assert len(port.calls) == 1
    tool_span, parent_span = exporter.get_finished_spans()
    assert tool_span.name == "execute_tool subagent"
    assert tool_span.parent.span_id == parent_span.context.span_id
    assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool_span.attributes["gen_ai.tool.call.id"] == "call"
    assert json.loads(tool_span.attributes["gen_ai.tool.call.arguments"]) == prepared.arguments
    assert tool_span.status.status_code is (
        StatusCode.ERROR if status == "failed" else StatusCode.UNSET
    )
    if status == "waiting":
        assert outcome is port.outcome
        assert "gen_ai.tool.call.result" not in tool_span.attributes
    else:
        assert isinstance(outcome, ToolResult)
        assert outcome.tool_use_id == "call"
        assert outcome.is_error is (status == "failed")
        assert json.loads(tool_span.attributes["gen_ai.tool.call.result"]) == {
            "parts": [{"type": "text", "content": outcome.model_content}]
        }


@pytest.mark.asyncio
async def test_dedicated_executor_keeps_controller_failure_and_request(
    configured_tool: tuple[SubagentTool, RecordingPort],
    tmp_path: Path,
    observability: tuple[Observability, InMemorySpanExporter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation, exporter = observability
    tool, port = configured_tool
    original = IrisRunPersistenceError("child settlement failed")

    async def fail(invocation: SubagentInvocation) -> SubagentExecutionOutcome:
        raise original

    monkeypatch.setattr(port, "execute", fail)
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry, observability=observation)
    context = ToolExecutionContext(workspace_root=tmp_path)
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call", name="subagent", input={"prompt": "investigate"})], context
    ).calls[0]
    with pytest.raises(IrisRunPersistenceError) as caught:
        await executor.execute_subagent_prepared(
            prepared, context, parent_call=SubagentParentCall("parent", "call")
        )
    assert caught.value is original
    [span] = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert json.loads(span.attributes["gen_ai.tool.call.arguments"]) == prepared.arguments
    assert "gen_ai.tool.call.result" not in span.attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("selector,expected", [(None, "researcher"), ("reviewer", "reviewer")])
async def test_dedicated_call_resolves_once_and_preserves_parent_identity(
    configured_tool: tuple[SubagentTool, RecordingPort],
    tmp_path: Path,
    selector: str | None,
    expected: str,
) -> None:
    tool, port = configured_tool
    params = tool.validate_input({"prompt": "  investigate  ", "agent": selector})
    outcome = await tool.execute_subagent(
        params,
        ToolExecutionContext(workspace_root=tmp_path),
        parent_call=SubagentParentCall("parent", "call"),
    )
    assert isinstance(outcome, ToolResult)
    assert len(port.calls) == 1
    invocation = port.calls[0]
    assert invocation.parent_call == SubagentParentCall("parent", "call")
    assert invocation.call.prompt == "investigate"
    assert invocation.call.route.selector == expected
    assert invocation.call.route.config_path == tmp_path / f"{expected}.yaml"
    assert invocation.cancellation is None


@pytest.mark.asyncio
async def test_direct_arun_requires_dedicated_executor(
    configured_tool: tuple[SubagentTool, RecordingPort],
    tmp_path: Path,
) -> None:
    tool, port = configured_tool
    with pytest.raises(
        IrisToolExecutionError, match="SubagentTool requires the dedicated executor"
    ):
        await tool.arun({"prompt": "x"}, ToolExecutionContext(workspace_root=tmp_path))
    assert port.calls == []
