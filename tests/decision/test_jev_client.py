"""Jev HTTP 协议、期限和自有连接的精确测试。"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx2
import pytest

from iris.decision import (
    BooleanAnswer,
    BooleanQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    JevClient,
    ScoreAnswer,
    ScoreQuestion,
    jev,
)
from iris.exceptions import IrisDecisionError


def _request() -> DecisionRequest:
    return DecisionRequest(
        state={"text": "需要处理退款并确认原因"},
        questions={
            "route": ChoiceQuestion(
                instructions="选择负责团队", options={"billing": "退款", "technical": None}
            ),
            "urgent": BooleanQuestion(instructions="是否紧急？"),
            "quality": ScoreQuestion(instructions="信息完整度", levels=("无信息", "部分", "完整")),
        },
    )


def _response() -> dict[str, Any]:
    return {
        "model": "jev-1.13.0-actual",
        "answers": {
            "quality": {
                "type": "score",
                "score": 1.98,
                "probabilities": {"2": 0.99, "0": 0.0, "1": 0.01},
                "legend": {"2": "完整", "0": "无信息", "1": "部分"},
                "confidence": 0.95,
            },
            "urgent": {"type": "noul", "noul": 0.7},
            "route": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"technical": 0.2, "billing": 0.8},
                "confidence": 0.6,
            },
        },
        "usage": {"input_tokens": 40, "output_tokens": 12},
    }


def _mock_http(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx2.Request], httpx2.Response | Awaitable[httpx2.Response]],
) -> tuple[list[httpx2.AsyncClient], list[dict[str, Any]]]:
    clients: list[httpx2.AsyncClient] = []
    constructor_options: list[dict[str, Any]] = []
    original = httpx2.AsyncClient

    def create(**kwargs: Any) -> httpx2.AsyncClient:
        constructor_options.append(kwargs)
        client = original(transport=httpx2.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(jev.httpx2, "AsyncClient", create)
    return clients, constructor_options


@pytest.mark.asyncio
async def test_mixed_questions_use_one_post_and_preserve_answer_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=_response())

    clients, options = _mock_http(monkeypatch, handle)
    client = JevClient(api_key="test-key", model="jev-custom")
    assert clients == []
    assert jev.JEV_DEFAULT_MODEL == "jev-1.13.0"
    assert jev.JEV_DEFAULT_TIMEOUT_SECONDS == 5

    async with client:
        assert clients == []
        response = await client.evaluate(_request())

    assert clients[0].is_closed
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "https://api.typesafe.ai/v1/systemone"
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    assert options == [{"timeout": None}]
    assert json.loads(requests[0].content) == {
        "model": "jev-custom",
        "state": {"text": "需要处理退款并确认原因"},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "选择负责团队",
                "criteria": {"billing": "退款", "technical": None},
            },
            "urgent": {"type": "noul", "instructions": "是否紧急？"},
            "quality": {
                "type": "score",
                "instructions": "信息完整度",
                "criteria": ["无信息", "部分", "完整"],
            },
        },
    }
    assert response.provider == "typesafe"
    assert response.model == "jev-1.13.0-actual"
    assert response.usage.input_tokens == 40
    assert response.usage.output_tokens == 12
    assert response.answers["route"] == ChoiceAnswer(
        choice="billing", probabilities={"technical": 0.2, "billing": 0.8}, confidence=0.6
    )
    assert response.answers["urgent"] == BooleanAnswer(probability=0.7)
    assert response.answers["quality"] == ScoreAnswer(
        score=1.98,
        probabilities={2: 0.99, 0: 0.0, 1: 0.01},
        levels=("无信息", "部分", "完整"),
        confidence=0.95,
    )


@pytest.mark.asyncio
async def test_http_resource_is_reused_and_explicit_close_is_repeatable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients, _ = _mock_http(monkeypatch, lambda _: httpx2.Response(200, json=_response()))
    unused = JevClient(api_key="test-key")
    await unused.aclose()
    assert clients == []
    client = JevClient(api_key="test-key")
    await client.evaluate(_request())
    await client.evaluate(_request())
    assert len(clients) == 1
    await client.aclose()
    await client.aclose()
    assert clients[0].is_closed


@pytest.mark.parametrize(
    "options",
    [
        {"api_key": " "},
        {"api_key": "test-key", "model": " "},
        {"api_key": "test-key", "timeout_seconds": 0},
        {"api_key": "test-key", "timeout_seconds": -1},
        {"api_key": "test-key", "timeout_seconds": float("inf")},
        {"api_key": "test-key", "timeout_seconds": float("nan")},
    ],
)
def test_invalid_client_settings_fail_at_construction(options: dict[str, Any]) -> None:
    with pytest.raises(IrisDecisionError):
        JevClient(**options)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        ChoiceQuestion(instructions="选择", options={str(i): None for i in range(256)}),
        ScoreQuestion(instructions="评价", levels=tuple(str(i) for i in range(11))),
    ],
)
async def test_vendor_question_limits_fail_before_creating_http_resource(
    monkeypatch: pytest.MonkeyPatch, question: ChoiceQuestion | ScoreQuestion
) -> None:
    clients, _ = _mock_http(monkeypatch, lambda _: httpx2.Response(200, json=_response()))
    async with JevClient(api_key="test-key") as client:
        with pytest.raises(IrisDecisionError):
            await client.evaluate(DecisionRequest(state="材料", questions={"q": question}))
    assert clients == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("answers",), {}),
        (("answers", "extra"), {"type": "noul", "noul": 0.5}),
        (
            ("answers", "urgent"),
            {"type": "choice", "choice": "yes", "probabilities": {"yes": 1}, "confidence": 1},
        ),
        (("answers", "route", "choice"), "missing"),
        (("answers", "route", "probabilities"), {"billing": 1}),
        (("answers", "route", "probabilities"), {"billing": 0.8, "missing": 0.2}),
        (("answers", "urgent", "noul"), 1.1),
        (("answers", "route", "probabilities"), {"billing": 1.1, "technical": -0.1}),
        (("answers", "route", "confidence"), -0.1),
        (("answers", "quality", "score"), 2.1),
        (("answers", "quality", "probabilities"), {"0": 0, "1": 0, "3": 1}),
        (("answers", "quality", "legend"), {"0": "完整", "1": "部分", "2": "无信息"}),
        (("answers", "urgent", "noul"), "0.7"),
        (("usage", "input_tokens"), -1),
        (("model",), " "),
    ],
)
async def test_untrusted_response_is_rejected_once(
    monkeypatch: pytest.MonkeyPatch, path: tuple[str, ...], value: Any
) -> None:
    payload = _response()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    calls = 0

    def handle(_: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(200, json=payload)

    clients, _ = _mock_http(monkeypatch, handle)
    async with JevClient(api_key="test-key") as client:
        with pytest.raises(IrisDecisionError) as caught:
            await client.evaluate(_request())
    assert calls == 1
    assert caught.value.__cause__ is not None
    assert clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500])
async def test_http_failure_has_cause_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    calls = 0

    def handle(_: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(status, json={"message": "failure"})

    clients, _ = _mock_http(monkeypatch, handle)
    with pytest.raises(IrisDecisionError) as caught:
        async with JevClient(api_key="test-key", model="jev-custom") as client:
            await client.evaluate(_request())
    assert calls == 1
    assert clients[0].is_closed
    assert isinstance(caught.value.__cause__, httpx2.HTTPStatusError)
    assert caught.value.context == {"provider": "typesafe", "model": "jev-custom"}
    assert "test-key" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["network", "json"])
async def test_network_and_json_failure_have_cause_without_retry(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    calls = 0

    def handle(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        if failure == "network":
            raise httpx2.ConnectError("connection failed", request=request)
        return httpx2.Response(200, content=b"not json")

    _mock_http(monkeypatch, handle)
    async with JevClient(api_key="test-key") as client:
        with pytest.raises(IrisDecisionError) as caught:
            await client.evaluate(_request())
    assert calls == 1
    assert caught.value.__cause__ is not None


@pytest.mark.asyncio
async def test_total_deadline_cancels_http_and_raises_decision_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = asyncio.Event()
    calls = 0

    async def handle(_: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("请求只能被取消")

    _mock_http(monkeypatch, handle)
    async with JevClient(api_key="test-key", timeout_seconds=0.01) as client:
        with pytest.raises(IrisDecisionError) as caught:
            await client.evaluate(_request())
    assert calls == 1
    assert cancelled.is_set()
    assert isinstance(caught.value.__cause__, TimeoutError)


@pytest.mark.asyncio
async def test_outer_cancellation_propagates_to_http_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    calls = 0

    async def handle(_: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("请求只能被取消")

    clients, _ = _mock_http(monkeypatch, handle)
    async with JevClient(api_key="test-key") as client:
        task = asyncio.create_task(client.evaluate(_request()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert calls == 1
    assert cancelled.is_set()
    assert clients[0].is_closed
