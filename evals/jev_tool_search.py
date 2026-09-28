"""用 Jev 重排 Iris 工具候选的独立实验，从仓库根目录以模块方式运行。"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Annotated, Any, Literal

import httpx2
from pydantic import BaseModel, Field, TypeAdapter

from iris.config import init_config
from iris.exceptions import IrisConfigError, IrisProviderError
from iris.tools import BaseTool, ToolDefinition
from iris.tools.builtin import FILE_TOOL_CLASSES, AskQuestionTool, WebFetchTool, WebSearchTool
from iris.tools.discovery import DeferredToolIndex

from ._tool_search_metrics import selection_metrics
from ._typesafe import post_system_one

MODEL = "jev-1.13.0"
SHORTLIST = 5
THRESHOLD = 0.5
INPUT_USD_PER_MILLION = 0.042
CASES_PATH = Path(__file__).parent / "fixtures" / "jev_tool_search.json"
NO_TOOL = "__none__"
CHOICE_INSTRUCTIONS = (
    "Which single tool directly performs the operation explicitly requested in state.query? "
    "Select __none__ when no tool is needed or none of the listed tools directly fits. "
    "Do not choose a tool merely because it is on the same topic."
)


class SearchCase(BaseModel):
    """从本地题集解析的一条带人工参考答案的查询。"""

    id: str
    language: Literal["zh", "en"]
    query: str
    expected: list[str]


class _NoulAnswer(BaseModel):
    """TypeSafe 返回的单个语义匹配概率。"""

    type: Literal["noul"]
    noul: float = Field(ge=0, le=1)


class _Usage(BaseModel):
    """API 报告的实际 token 用量。"""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class _Response(BaseModel):
    """在 HTTP 边界解析一次的 Jev 响应。"""

    model: str
    answers: dict[str, _NoulAnswer]
    usage: _Usage


class _ChoiceAnswer(BaseModel):
    """单道 Choice 的首选、概率分布和置信度。"""

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Annotated[float, Field(ge=0, le=1)]]
    confidence: float = Field(ge=0, le=1)


class _ChoiceAnswers(BaseModel):
    """本实验固定只提出一道名为 selection 的选择题。"""

    selection: _ChoiceAnswer


class _ChoiceResponse(BaseModel):
    """从 TypeSafe API 解析的单选响应与用量。"""

    model: str
    answers: _ChoiceAnswers
    usage: _Usage


def load_catalog() -> list[BaseTool]:
    """构造八个真实内置工具的定义，只用于检索，不执行工具。"""
    tools: list[BaseTool] = [tool_class() for tool_class in FILE_TOOL_CLASSES]
    tools.extend([WebSearchTool(api_key=""), WebFetchTool(api_key=""), AskQuestionTool()])
    for tool in tools:
        tool.definition = tool.definition.model_copy(update={"deferred": True})
    return tools


def _tool_summary(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "group": tool.group,
        "tags": tool.metadata.get("tags", []),
    }


async def evaluate_case(
    client: httpx2.AsyncClient,
    api_key: str,
    case: SearchCase,
    index: DeferredToolIndex,
    *,
    catalog: list[ToolDefinition],
    pool: Literal["bm25", "all"],
) -> dict[str, Any]:
    """比较本地搜索与 Jev 选择，保存候选、原始响应和耗时。

    Args:
        client: 整个实验复用的 HTTP client。
        api_key: 从 Iris 配置读取的 TypeSafe 凭据，不进入结果。
        case: 本地题目；参考答案只在本地计分。
        index: 由同一工具目录构建的当前 Iris 搜索索引。
        catalog: all 模式下使用的完整实验工具目录。
        pool: bm25 只重排初筛候选；all 诊断词法召回损失。

    Returns:
        可直接写入 JSON 的单题实验记录。

    Raises:
        IrisProviderError: HTTP 请求失败或外部响应不符合当前契约。
    """
    start = perf_counter()
    baseline = index.search(case.query, limit=SHORTLIST)
    baseline_ms = (perf_counter() - start) * 1000
    candidates = baseline if pool == "bm25" else catalog
    row: dict[str, Any] = {
        **case.model_dump(),
        "baseline": [tool.name for tool in baseline[:3]],
        "candidates": [tool.name for tool in candidates],
        "baseline_ms": baseline_ms,
        "jev": [],
        "jev_ms": None,
        "model": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "response": None,
    }
    if not candidates:
        return row
    start = perf_counter()
    parsed = await query_jev(client, api_key, case.query, candidates)
    scores = {tool.name: parsed.answers[tool.name].noul for tool in candidates}
    # 同分保留候选原顺序；阈值预先固定，不根据实验答案调参。
    ranked = sorted(scores, key=lambda name: scores[name], reverse=True)
    row.update(
        jev=[name for name in ranked if scores[name] >= THRESHOLD][:3],
        jev_ms=(perf_counter() - start) * 1000,
        model=parsed.model,
        input_tokens=parsed.usage.input_tokens,
        output_tokens=parsed.usage.output_tokens,
        response=parsed.model_dump(),
    )
    return row


async def query_jev(
    client: httpx2.AsyncClient, api_key: str, query: str, candidates: list[ToolDefinition]
) -> _Response:
    """向 Jev 批量提交同一查询的工具匹配问题，供两个实验入口复用。"""
    payload = {
        "model": MODEL,
        "state": {"query": query},
        "questions": {
            tool.name: {
                "type": "noul",
                "instructions": {
                    "tool": _tool_summary(tool),
                    "question": (
                        "Is this tool a direct match for the operation explicitly requested "
                        "in state.query? Judge the tool's described capability, not word overlap."
                    ),
                },
                "criteria": {
                    "true": "The request asks for an operation this tool directly performs.",
                    "false": (
                        "The tool is only topically related, the operation is excluded, "
                        "or the request needs no tool."
                    ),
                },
            }
            for tool in candidates
        },
    }
    parsed = await post_system_one(client, api_key, payload, _Response)
    if set(parsed.answers) != {tool.name for tool in candidates}:
        raise IrisProviderError("TypeSafe 响应的问题集合与请求不一致")
    return parsed


async def evaluate_choice(
    client: httpx2.AsyncClient, api_key: str, query: str, candidates: list[ToolDefinition]
) -> dict[str, Any]:
    """用一道 Choice 选择最佳工具或无需工具，不使用 Noul 阈值。

    Args:
        client: 实验复用的 HTTP client。
        api_key: 由 Iris 配置读取的 TypeSafe 凭据。
        query: 不含参考答案的用户查询。
        candidates: 每个工具作为一个选项，元数据只提交一次。

    Returns:
        零个或一个工具、完整概率、置信度、请求与响应及实际用量。

    Raises:
        IrisProviderError: 请求、响应或首选名称不符合当前契约。
    """
    criteria: dict[str, Any] = {tool.name: _tool_summary(tool) for tool in candidates}
    criteria[NO_TOOL] = (
        "No tool is needed, the operation is excluded, or none of the listed tools directly fits."
    )
    payload = {
        "model": MODEL,
        "state": {"query": query},
        "questions": {
            "selection": {
                "type": "choice",
                "instructions": CHOICE_INSTRUCTIONS,
                "criteria": criteria,
            }
        },
    }
    response = await post_system_one(client, api_key, payload, _ChoiceResponse)
    answer = response.answers.selection
    if answer.choice not in criteria:
        raise IrisProviderError("TypeSafe Choice 返回了候选集合之外的选项")
    return {
        "selected": [] if answer.choice == NO_TOOL else [answer.choice],
        "choice_probabilities": answer.probabilities,
        "choice_confidence": answer.confidence,
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "request": payload,
        "response": response.model_dump(),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """分别统计正例命中、负例误选和实际 API 用量，空分母记为 null。"""
    metrics = selection_metrics(rows, "baseline")
    summary: dict[str, Any] = {"cases": len(rows), **metrics}
    for key in ("top1", "top3", "false_selection"):
        summary.pop(key)
    for method in ("baseline", "jev"):
        metrics = selection_metrics(rows, method)
        for key in ("top1", "top3", "false_selection"):
            summary[f"{method}_{key}"] = metrics[key]
    latency = sorted(row["jev_ms"] for row in rows if row["jev_ms"] is not None)
    summary.update(
        api_calls=len(latency),
        baseline_median_ms=median(row["baseline_ms"] for row in rows),
        jev_median_ms=median(latency) if latency else None,
        jev_p95_ms=latency[math.ceil(len(latency) * 0.95) - 1] if latency else None,
        input_tokens=sum(row["input_tokens"] for row in rows),
        output_tokens=sum(row["output_tokens"] for row in rows),
    )
    summary["estimated_input_usd"] = summary["input_tokens"] * INPUT_USD_PER_MILLION / 1_000_000
    return summary


async def main() -> None:
    """读取既有 Iris 配置并执行有真实 API 费用的独立实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--pool", choices=("bm25", "all"), default="bm25")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = init_config(env_file=args.env_file)
    api_key = config.provider_api_keys.get("typesafe")
    if not api_key:
        raise IrisConfigError("缺少 IRIS_PROVIDER_API_KEYS__TYPESAFE")
    tools = load_catalog()
    catalog = [tool.definition for tool in tools]
    index = DeferredToolIndex()
    index.build(tools)
    cases = TypeAdapter(list[SearchCase]).validate_json(CASES_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    output = args.output or Path(f"tmp/jev-tool-search-{args.pool}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "dataset": "handwritten-bilingual-v1",
        "model": MODEL,
        "pool": args.pool,
        "shortlist": SHORTLIST,
        "threshold": THRESHOLD,
        "input_usd_per_million": INPUT_USD_PER_MILLION,
        "catalog": [_tool_summary(tool) for tool in catalog],
        "rows": rows,
    }
    async with httpx2.AsyncClient(timeout=30.0) as client:
        for case in cases:
            row = await evaluate_case(client, api_key, case, index, catalog=catalog, pool=args.pool)
            rows.append(row)
            # 每题保存已完成结果，中途 API 失败仍能核对已产生的用量。
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"{len(rows)}/{len(cases)} {case.id}: {row['baseline']} -> {row['jev']}", flush=True
            )
    report["summary"] = summarize(rows)
    report["by_language"] = {
        language: summarize([row for row in rows if row["language"] == language])
        for language in ("zh", "en")
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"结果已保存：{output.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
