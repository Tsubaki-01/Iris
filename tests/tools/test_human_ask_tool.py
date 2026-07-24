from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from iris.exceptions import IrisHITLError
from iris.hitl.models import QuestionPrompt, ToolCallSnapshot, make_call_fingerprint
from iris.hitl.tools import AskQuestionInput, AskQuestionTool
from iris.message import ToolUseBlock
from iris.tools import ToolExecutionContext, ToolExecutor, ToolRegistry


def test_ask_question_tool_exposes_human_question_schema() -> None:
    tool = AskQuestionTool()

    assert tool.definition.name == "ask_question"
    assert tool.definition.capabilities == set()
    assert tool.definition.group == "human"
    assert tool.definition.input_schema["required"] == ["question"]


@pytest.mark.parametrize(
    "value",
    [
        {"question": "  "},
        {"question": "继续吗？", "options": ["继续", "  "]},
        {"question": "继续吗？", "options": ["继续", " 继续 "]},
    ],
)
def test_ask_question_input_rejects_empty_or_duplicate_values(value: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AskQuestionInput.model_validate(value)


def test_ask_question_tool_builds_question_prompt() -> None:
    prompt = AskQuestionTool().build_interaction_prompt(
        params={"question": "是否继续？", "options": ["继续", "取消"]},
    )

    assert isinstance(prompt, QuestionPrompt)
    assert prompt.question == "是否继续？"
    assert prompt.options == ["继续", "取消"]


def test_preflight_classifies_ask_question_as_a_human_gate() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())

    context = ToolExecutionContext(
        workspace_root=".",
        session_id="session_1",
        metadata={"run_id": "run_1"},
    )
    prepared = (
        ToolExecutor(registry)
        .prepare_many(
            [ToolUseBlock(id="call_1", name="ask_question", input={"question": "继续吗？"})],
            context,
        )
        .calls[0]
    )

    assert prepared.human_request is not None
    assert isinstance(prepared.human_request.prompt, QuestionPrompt)
    assert prepared.human_request.tool_call == ToolCallSnapshot(
        tool_call_id="call_1",
        tool_name="ask_question",
        arguments={"question": "继续吗？", "options": []},
        workspace_root=str(context.workspace_root.resolve()),
        fingerprint=make_call_fingerprint(
            session_id="session_1",
            run_id="run_1",
            tool_call_id="call_1",
            tool_name="ask_question",
            arguments={"question": "继续吗？", "options": []},
            workspace_root=str(context.workspace_root.resolve()),
        ),
    )


def test_ask_question_tool_rejects_direct_execution() -> None:
    tool = AskQuestionTool()

    with pytest.raises(IrisHITLError):
        asyncio.run(
            tool.arun(
                {"question": "是否继续？"},
                ToolExecutionContext(workspace_root="."),
            )
        )
