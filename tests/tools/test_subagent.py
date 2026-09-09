"""SubagentTool 的模型输入与专用调用契约。"""

from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from iris.exceptions import IrisToolExecutionError, IrisToolValidationError
from iris.hitl.models import HumanInteraction
from iris.tools.base import ToolCapability, ToolExecutionContext, ToolResult
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


@pytest.mark.asyncio
async def test_waiting_outcome_and_cancellation_flow_through_dedicated_port(
    configured_tool: tuple[SubagentTool, RecordingPort],
    tmp_path: Path,
) -> None:
    tool, port = configured_tool
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
