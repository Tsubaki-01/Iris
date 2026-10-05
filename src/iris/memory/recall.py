"""在允许的完整记忆候选中，通过一次 Score 请求直接语义召回。"""

from collections.abc import Sequence
from typing import Any, cast

from ..decision import DecisionEvaluator, DecisionRequest, ScoreAnswer, ScoreQuestion
from ..exceptions import IrisDecisionError, IrisMemoryError
from ..prompts import PromptSnapshot
from ._prompts import render_memory_prompt
from ._query import matches_required_phrases
from .models import MemoryItem, MemorySearchHit, MemorySearchQuery, MemorySearchResponse

_RELEVANCE_LEVELS = (
    "与问题无关，或正文明确的适用条件不成立。",
    "只有同主题背景，没有可用于回答的具体依据。",
    "提供部分可用回答依据，不足以独立回答全部问题。",
    "直接提供所需回答依据，正文可见的适用条件成立。",
)


async def recall_memories(
    candidates: Sequence[MemoryItem],
    query: MemorySearchQuery,
    evaluator: DecisionEvaluator,
    prompt_snapshot: PromptSnapshot | None,
) -> tuple[MemorySearchResponse, dict[str, Any]]:
    """筛选显式必要词组，评分全部剩余正文并返回完整原文及独立用量。"""
    eligible = [
        item for item in candidates if matches_required_phrases(item.text, query.required_terms)
    ]
    if not eligible:
        return MemorySearchResponse((), False), {}
    if prompt_snapshot is None:
        raise IrisMemoryError("memory recall prompt 来源未配置")
    request = DecisionRequest.model_construct(
        state={"query": query.query, "memories": [item.text for item in eligible]},
        questions={
            f"m{i}": ScoreQuestion.model_construct(
                instructions=render_memory_prompt(
                    prompt_snapshot, "memory_recall_instruction", {"candidate_index": i}
                ),
                levels=_RELEVANCE_LEVELS,
            )
            for i in range(len(eligible))
        },
    )
    try:
        response = await evaluator.evaluate(request)
    except IrisDecisionError as exc:
        raise IrisMemoryError("记忆 Decision 召回失败", error=str(exc)) from exc
    scored: list[tuple[float, MemoryItem]] = []
    for i, item in enumerate(eligible):
        answer = cast(ScoreAnswer, response.answers[f"m{i}"])
        if answer.score >= 2.0:
            scored.append((answer.score, item))
    scored.sort(key=lambda pair: -pair[0])
    hits = tuple(
        MemorySearchHit(item.id, item.namespace, item.category, item.kind, item.text, True)
        for _, item in scored[: query.limit]
    )
    return MemorySearchResponse(hits, len(scored) > query.limit), {
        "decision": {
            "feature": "memory.recall",
            "provider": response.provider,
            "model": response.model,
            "question_count": len(eligible),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    }
