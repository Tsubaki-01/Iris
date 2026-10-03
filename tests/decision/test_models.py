"""Decision 的公开请求边界与可信结果模型。"""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from pydantic import ValidationError

from iris.decision.models import (
    BooleanAnswer,
    BooleanQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    ScoreAnswer,
    ScoreQuestion,
)


def test_request_parses_three_question_types_with_shared_state() -> None:
    """原始 SDK 数据只在请求模型边界解析为封闭题型。"""
    request = DecisionRequest.model_validate(
        {
            "state": {"query": "read", "candidates": ["reader", "writer"]},
            "questions": {
                "pick": {
                    "type": "choice",
                    "instructions": "Pick a tool.",
                    "options": {"reader": None, "none": "No match"},
                },
                "check": {"type": "boolean", "instructions": "A reader is available."},
                "grade": {
                    "type": "score",
                    "instructions": "Rate relevance.",
                    "levels": ["unrelated", "related"],
                },
            },
        }
    )
    assert isinstance(request.questions["pick"], ChoiceQuestion)
    assert isinstance(request.questions["check"], BooleanQuestion)
    assert isinstance(request.questions["grade"], ScoreQuestion)
    assert request.questions["grade"].levels == ("unrelated", "related")
    assert request.model_dump(mode="json")["state"] == request.state
    with pytest.raises(ValidationError):
        request.state = "changed"


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "pick", "options": {}},
        {"type": "choice", "instructions": "pick", "options": {" ": None}},
        {"type": "boolean", "instructions": " \n"},
        {"type": "score", "instructions": "rate", "levels": ["one"]},
        {"type": "score", "instructions": "rate", "levels": ["one", " "]},
        {"type": "noul", "instructions": "check"},
        {"type": "boolean", "instructions": "check", "unknown": True},
    ],
)
def test_invalid_raw_question_is_rejected(question: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate({"state": None, "questions": {"q": question}})


@pytest.mark.parametrize("questions", [{}, {" ": BooleanQuestion(instructions="Check.")}])
def test_request_requires_nonempty_question_ids_and_questions(questions: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DecisionRequest(state=None, questions=questions)


def test_request_rejects_unknown_fields_and_non_json_state() -> None:
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(
            {
                "state": None,
                "questions": {"q": BooleanQuestion(instructions="Check.")},
                "model": "jev",
            }
        )
    with pytest.raises(ValidationError):
        DecisionRequest(state=object(), questions={"q": BooleanQuestion(instructions="Check.")})


def test_trusted_response_preserves_service_scores_and_usage() -> None:
    """消费者直接使用已解析答案，不自行重算或校验分布。"""
    choice = ChoiceAnswer(choice="reader", probabilities={"reader": 1.0}, confidence=1.0)
    boolean = BooleanAnswer(probability=0.8)
    score = ScoreAnswer(
        score=0.98,
        probabilities={0: 0.01, 1: 0.99},
        levels=("unrelated", "related"),
        confidence=0.9,
    )
    response = DecisionResponse(
        provider="typesafe",
        model="jev-1.13.0",
        answers={"pick": choice, "check": boolean, "grade": score},
        usage=DecisionUsage(input_tokens=12, output_tokens=3),
    )
    assert (choice.type, boolean.type, score.type) == ("choice", "boolean", "score")
    assert score.score == 0.98
    assert response.answers["grade"] is score
    assert response.usage.input_tokens == 12
    with pytest.raises(FrozenInstanceError):
        response.model = "changed"  # type: ignore[misc]
