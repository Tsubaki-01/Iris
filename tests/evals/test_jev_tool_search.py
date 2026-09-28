"""验证 Jev 实验的真实请求形状、排序和评分口径，不调用外网。"""

import json
from typing import Any

import httpx2
import pytest

from evals.jev_tool_search import (
    NO_TOOL,
    SearchCase,
    evaluate_case,
    evaluate_choice,
    load_catalog,
    summarize,
)
from iris.exceptions import IrisProviderError
from iris.tools import ToolDefinition
from iris.tools.discovery import DeferredToolIndex


def _response(scores: dict[str, float]) -> dict[str, Any]:
    return {
        "model": "jev-1.13.0",
        "answers": {name: {"type": "noul", "noul": score} for name, score in scores.items()},
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }


def _index() -> DeferredToolIndex:
    tools = load_catalog()
    index = DeferredToolIndex()
    index.build(tools)
    return index


@pytest.mark.asyncio
async def test_rerank_uses_candidates_and_never_sends_expected_labels() -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx2.Response(
            200,
            json=_response(
                {name: 0.95 if name == "grep_search" else 0.1 for name in payload["questions"]}
            ),
        )

    case = SearchCase(id="search", language="zh", query="搜索文本文件", expected=["grep_search"])
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        row = await evaluate_case(client, "test-key", case, _index(), catalog=[], pool="bm25")
    payload = requests[0]
    assert payload["state"] == {"query": case.query}
    assert set(payload["questions"]) == set(row["candidates"])
    assert "expected" not in json.dumps(payload)
    assert all(question["type"] == "noul" for question in payload["questions"].values())
    assert row["jev"] == ["grep_search"]
    assert row["input_tokens"] == 100
    assert row["model"] == "jev-1.13.0"


@pytest.mark.asyncio
async def test_empty_shortlist_skips_api_instead_of_expanding_candidates() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        pytest.fail("空候选不应调用 API")

    case = SearchCase(id="miss", language="en", query="xyzzy", expected=["read_file"])
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        row = await evaluate_case(client, "test-key", case, _index(), catalog=[], pool="bm25")
    assert row["candidates"] == row["jev"] == []
    assert row["input_tokens"] == 0
    assert row["jev_ms"] is None
    summary = summarize([row])
    assert summary["candidate_recall"] == 0
    assert summary["jev_top1"] == 0


@pytest.mark.asyncio
async def test_full_catalog_can_retrieve_lexically_missing_tool() -> None:
    tool = ToolDefinition(name="read_file", description="读取文件", input_schema={"type": "object"})

    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_response({"read_file": 0.9}))

    case = SearchCase(id="semantic", language="en", query="xyzzy", expected=["read_file"])
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        row = await evaluate_case(client, "test-key", case, _index(), catalog=[tool], pool="all")
    assert row["baseline"] == []
    assert row["jev"] == ["read_file"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status,scores", [(401, {}), (200, {"read_file": 1.5})])
async def test_api_failure_is_not_reported_as_baseline_success(
    status: int, scores: dict[str, float]
) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=_response(scores))

    case = SearchCase(id="bad", language="en", query="read_file", expected=["read_file"])
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        with pytest.raises(IrisProviderError):
            await evaluate_case(client, "test-key", case, _index(), catalog=[], pool="bm25")


def test_metrics_separate_positive_recall_and_negative_false_selection() -> None:
    common = {"baseline_ms": 1.0, "jev_ms": 100.0, "input_tokens": 100, "output_tokens": 10}
    rows = [
        dict(common, expected=["a"], candidates=["b", "a"], baseline=["b", "a"], jev=["a"]),
        dict(common, expected=["c"], candidates=["b"], baseline=["b"], jev=[]),
        dict(common, expected=[], candidates=["b"], baseline=["b"], jev=[]),
    ]
    summary = summarize(rows)
    assert summary["candidate_recall"] == 0.5
    assert summary["baseline_top1"] == 0
    assert summary["baseline_top3"] == 0.5
    assert summary["jev_top1"] == 0.5
    assert summary["jev_top3"] == 0.5
    assert summary["baseline_false_selection"] == 1
    assert summary["jev_false_selection"] == 0
    assert summary["api_calls"] == 3
    assert summary["input_tokens"] == 300


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choice,probabilities,expected",
    [
        ("read_file", {"read_file": 0.4, "write_file": 0.35, "__none__": 0.25}, ["read_file"]),
        ("__none__", {"read_file": 0.3, "write_file": 0.3, "__none__": 0.4}, []),
    ],
)
async def test_choice_uses_one_question_and_does_not_apply_noul_threshold(
    choice: str, probabilities: dict[str, float], expected: list[str]
) -> None:
    tools = [
        ToolDefinition(name="read_file", description="读取文本", input_schema={"type": "object"}),
        ToolDefinition(name="write_file", description="写入文本", input_schema={"type": "object"}),
    ]

    def respond(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        assert payload["state"] == {"query": "读取说明"}
        assert list(payload["questions"]) == ["selection"]
        question = payload["questions"]["selection"]
        assert question["type"] == "choice"
        assert set(question["criteria"]) == {"read_file", "write_file", NO_TOOL}
        assert question["criteria"]["read_file"]["description"] == "读取文本"
        assert "expected" not in payload
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "selection": {
                        "type": "choice",
                        "choice": choice,
                        "probabilities": probabilities,
                        "confidence": 0.1,
                    }
                },
                "usage": {"input_tokens": 200, "output_tokens": 30},
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        result = await evaluate_choice(client, "test-key", "读取说明", tools)
    assert result["selected"] == expected
    assert result["choice_probabilities"] == probabilities
    assert result["choice_confidence"] == 0.1
    assert result["input_tokens"] == 200


@pytest.mark.asyncio
async def test_choice_rejects_an_option_outside_the_requested_set() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "selection": {
                        "type": "choice",
                        "choice": "unknown",
                        "probabilities": {"unknown": 1.0},
                        "confidence": 1.0,
                    }
                },
                "usage": {"input_tokens": 200, "output_tokens": 30},
            },
        )

    tools = [tool.definition for tool in load_catalog()]
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        with pytest.raises(IrisProviderError):
            await evaluate_choice(client, "test-key", "读取说明", tools)
