"""Decision 公开请求边界与解析后可信的判断结果。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

_NonBlank = Annotated[str, StringConstraints(pattern=r"\S")]


class _Question(BaseModel):
    """问题原始输入共用的校验边界。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instructions: _NonBlank


class ChoiceQuestion(_Question):
    """从给定选项中选择唯一一项。"""

    type: Literal["choice"] = "choice"
    options: dict[_NonBlank, str | None] = Field(min_length=1)


class BooleanQuestion(_Question):
    """判断一个命题成立的概率。"""

    type: Literal["boolean"] = "boolean"


class ScoreQuestion(_Question):
    """按照从低到高的描述档位评分。"""

    type: Literal["score"] = "score"
    levels: tuple[_NonBlank, ...] = Field(min_length=2)


DecisionQuestion = Annotated[
    ChoiceQuestion | BooleanQuestion | ScoreQuestion, Field(discriminator="type")
]


class DecisionRequest(BaseModel):
    """一次共享状态上的独立问题集合，不携带生成式消息或工具协议。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    state: JsonValue
    questions: dict[_NonBlank, DecisionQuestion] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """选中选项、服务返回的分布与置信度。"""

    choice: str
    probabilities: dict[str, float]
    confidence: float
    type: Literal["choice"] = "choice"


@dataclass(frozen=True, slots=True)
class BooleanAnswer:
    """命题成立概率。"""

    probability: float
    type: Literal["boolean"] = "boolean"


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """服务返回的期望档位及其原始分布，不自行重算 score。"""

    score: float
    probabilities: dict[int, float]
    levels: tuple[str, ...]
    confidence: float
    type: Literal["score"] = "score"


DecisionAnswer = ChoiceAnswer | BooleanAnswer | ScoreAnswer


@dataclass(frozen=True, slots=True)
class DecisionUsage:
    """独立于主聊天模型的本次判断用量。"""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class DecisionResponse:
    """按原始问题 ID 定位的可信答案及实际后端信息。"""

    provider: str
    model: str
    answers: dict[str, DecisionAnswer]
    usage: DecisionUsage


__all__ = [
    "BooleanAnswer",
    "BooleanQuestion",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecisionAnswer",
    "DecisionQuestion",
    "DecisionRequest",
    "DecisionResponse",
    "DecisionUsage",
    "ScoreAnswer",
    "ScoreQuestion",
]
