"""Jev 的单请求 HTTP 实现及唯一外部响应解析边界。"""

import asyncio
import math
from types import TracebackType
from typing import Annotated, Any, Literal, Self

import httpx2
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, model_validator

from ..exceptions import IrisDecisionError
from .models import (
    BooleanAnswer,
    ChoiceAnswer,
    DecisionAnswer,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    ScoreAnswer,
)

JEV_DEFAULT_MODEL = "jev-1.13.0"
JEV_DEFAULT_TIMEOUT_SECONDS = 5.0
_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

_Probability = Annotated[float, Field(ge=0, le=1)]


class _WireModel(BaseModel):
    """Jev 外部 JSON 的字段类型及有限数值约束。"""

    model_config = ConfigDict(strict=True, allow_inf_nan=False)


class _Choice(_WireModel):
    """Jev Choice 原始答案。"""

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, _Probability]
    confidence: _Probability


class _Noul(_WireModel):
    """Jev Noul 原始答案。"""

    type: Literal["noul"]
    noul: _Probability


class _Score(_WireModel):
    """Jev Score 原始答案，保留厂商显示的数值。"""

    type: Literal["score"]
    score: float = Field(ge=0)
    probabilities: dict[str, _Probability]
    legend: dict[str, str]
    confidence: _Probability


class _Usage(_WireModel):
    """当前请求的独立 token 用量。"""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class _JevResponse(_WireModel):
    """一次解析结构、数值和请求对应关系的 Jev 响应。"""

    model: str = Field(pattern=r"\S")
    answers: dict[str, Annotated[_Choice | _Noul | _Score, Field(discriminator="type")]]
    usage: _Usage

    @model_validator(mode="after")
    def match_request(self, info: ValidationInfo) -> Self:
        """外部答案必须精确对应已发送的问题及选项。"""
        request: DecisionRequest = info.context
        if self.answers.keys() != request.questions.keys():
            raise ValueError("Jev 答案题号与请求不一致")
        for question_id, question in request.questions.items():
            answer = self.answers[question_id]
            if question.type == "choice":
                if answer.type != "choice":
                    raise ValueError("Jev Choice 答案类型错误")
                if (
                    answer.choice not in question.options
                    or answer.probabilities.keys() != question.options.keys()
                ):
                    raise ValueError("Jev Choice 答案选项与请求不一致")
            elif question.type == "boolean":
                if answer.type != "noul":
                    raise ValueError("Jev Boolean 答案类型错误")
            else:
                if answer.type != "score":
                    raise ValueError("Jev Score 答案类型错误")
                expected = {str(i): level for i, level in enumerate(question.levels)}
                if answer.legend != expected or answer.probabilities.keys() != expected.keys():
                    raise ValueError("Jev Score 答案等级与请求不一致")
                if answer.score > len(question.levels) - 1:
                    raise ValueError("Jev Score 超出等级范围")
        return self


def _request_payload(request: DecisionRequest, model: str) -> dict[str, Any]:
    """投影已验证请求，仅在厂商边界检查其容量上限。"""
    questions: dict[str, dict[str, Any]] = {}
    for question_id, question in request.questions.items():
        wire: dict[str, Any] = {"type": question.type, "instructions": question.instructions}
        if question.type == "choice":
            if len(question.options) > 255:
                raise IrisDecisionError(
                    "Jev Choice 最多支持 255 个选项", provider="typesafe", model=model
                )
            wire["criteria"] = question.options
        elif question.type == "boolean":
            wire["type"] = "noul"
        else:
            if len(question.levels) > 10:
                raise IrisDecisionError(
                    "Jev Score 最多支持 10 个等级", provider="typesafe", model=model
                )
            wire["criteria"] = list(question.levels)
        questions[question_id] = wire
    return {"model": model, "state": request.state, "questions": questions}


def _project_response(response: _JevResponse) -> DecisionResponse:
    """将已验证的厂商答案直接投影为公共可信对象。"""
    answers: dict[str, DecisionAnswer] = {}
    for question_id, answer in response.answers.items():
        if answer.type == "choice":
            answers[question_id] = ChoiceAnswer(
                choice=answer.choice,
                probabilities=answer.probabilities,
                confidence=answer.confidence,
            )
        elif answer.type == "noul":
            answers[question_id] = BooleanAnswer(probability=answer.noul)
        else:
            answers[question_id] = ScoreAnswer(
                score=answer.score,
                probabilities={int(key): value for key, value in answer.probabilities.items()},
                levels=tuple(answer.legend[str(i)] for i in range(len(answer.legend))),
                confidence=answer.confidence,
            )
    return DecisionResponse(
        provider="typesafe",
        model=response.model,
        answers=answers,
        usage=DecisionUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        ),
    )


class JevClient:
    """自有惰性 HTTP 连接的 Jev 客户端，每次评价只发送一次请求。"""

    def __init__(
        self,
        api_key: str,
        model: str = JEV_DEFAULT_MODEL,
        timeout_seconds: float = JEV_DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """保存凭据、模型及总期限，构造期间不创建 HTTP 资源。"""
        if not api_key.strip():
            raise IrisDecisionError("Jev API key 不能为空", provider="typesafe", model=model)
        if not model.strip():
            raise IrisDecisionError("Jev model 不能为空", provider="typesafe", model=model)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise IrisDecisionError("Jev 总期限必须为正有限数", provider="typesafe", model=model)
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._client: httpx2.AsyncClient | None = None

    async def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        """在一次总期限内发送混合问题并解析，外层取消原样传播。"""
        try:
            async with asyncio.timeout(self._timeout_seconds):
                payload = _request_payload(request, self._model)
                if self._client is None:
                    self._client = httpx2.AsyncClient(timeout=None)
                response = await self._client.post(
                    _JEV_ENDPOINT,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                parsed = _JevResponse.model_validate_json(response.content, context=request)
                return _project_response(parsed)
        except httpx2.HTTPStatusError as exc:
            raise IrisDecisionError(
                f"Jev HTTP {exc.response.status_code}", provider="typesafe", model=self._model
            ) from exc
        except (TimeoutError, httpx2.RequestError, ValidationError) as exc:
            raise IrisDecisionError(
                "Jev 请求超时、网络失败或响应解析失败", provider="typesafe", model=self._model
            ) from exc

    async def aclose(self) -> None:
        """关闭已创建的自有连接；未调用时无需创建资源。"""
        if self._client is not None:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        """进入异步上下文时继续保持连接惰性创建。"""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """正常或异常退出均释放自有连接。"""
        await self.aclose()
