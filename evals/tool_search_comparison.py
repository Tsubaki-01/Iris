"""在相同查询上对比规则、Jev Noul、Jev Choice 与当前配置的 DeepSeek。"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Literal

import httpx2
import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from iris.agents.config.base import ModelConfig
from iris.config import init_config
from iris.exceptions import IrisConfigError, IrisProviderError
from iris.providers import create_provider_client
from iris.tools import ToolDefinition
from iris.tools.discovery import DeferredToolIndex

from . import jev_tool_search as jev
from ._tool_search_metrics import selection_metrics

DEEPSEEK_PROMPT = (
    "Select up to three directly relevant tools, best first. Return JSON only: "
    '{"tools": ["tool_name"]}. Return an empty list if no tool is needed. '
    "Exclude tools that are only topically related."
)
DEEPSEEK_OFF_PEAK = {"cache_miss": 0.15, "output": 0.6}
METHODS = ("rules", "jev_noul", "jev_choice", "deepseek")


class _DeepSeekUsage(BaseModel):
    """官方接口的输入、输出及缓存计费桶。"""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    prompt_cache_hit_tokens: int = Field(ge=0)
    prompt_cache_miss_tokens: int = Field(ge=0)


class _Message(BaseModel):
    """用于选择工具的 JSON 文本。"""

    content: str


class _Choice(BaseModel):
    """仅接受完整结束的模型输出，截断输出不能算成功。"""

    message: _Message
    finish_reason: Literal["stop"]


class _DeepSeekResponse(BaseModel):
    """保留本次实际使用的 DeepSeek 外部响应字段。"""

    model: str
    choices: list[_Choice] = Field(min_length=1)
    usage: _DeepSeekUsage


class _Selection(BaseModel):
    """由生成模型返回的有限工具选择。"""

    tools: list[str] = Field(max_length=3)


async def evaluate_deepseek(
    client: httpx2.AsyncClient,
    api_key: str,
    model: ModelConfig,
    query: str,
    candidates: list[ToolDefinition],
    *,
    endpoint: str = "https://api.deepseek.com/chat/completions",
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """调用当前 DeepSeek 模型选择候选，保存原始缓存用量。

    Args:
        client: 整轮实验复用的 HTTP client。
        api_key: 由 Iris provider factory 解析的凭据。
        model: 当前 Agent YAML 中的模型配置。
        query: 不含参考答案的用户查询。
        candidates: 与 Jev 相同的候选工具元数据。
        endpoint: 当前 provider 配置解析出的 Chat Completions 地址。
        headers: 当前 provider 的附加 HTTP headers。

    Returns:
        选择、实际模型、token 计费桶和原始响应。

    Raises:
        IrisProviderError: API、JSON 或候选名称不符合当前契约。
    """
    payload = {
        "model": model.name,
        "messages": [
            {"role": "system", "content": DEEPSEEK_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"query": query, "tools": [jev._tool_summary(tool) for tool in candidates]},
                    ensure_ascii=False,
                ),
            },
        ],
        **{
            key: value
            for key, value in model.to_llm_request_options().items()
            if key in {"temperature", "top_p", "max_tokens"}
        },
        "response_format": {"type": "json_object"},
    }
    try:
        response = await client.post(
            endpoint,
            headers={**(headers or {}), "Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=model.timeout or 60.0,
        )
        response.raise_for_status()
        raw = response.json()
        parsed = _DeepSeekResponse.model_validate(raw)
        selection = _Selection.model_validate_json(parsed.choices[0].message.content)
    except httpx2.HTTPStatusError as exc:
        raise IrisProviderError(f"DeepSeek HTTP {exc.response.status_code}") from exc
    except (httpx2.RequestError, ValidationError, json.JSONDecodeError) as exc:
        raise IrisProviderError("DeepSeek 请求或选择解析失败") from exc
    if not set(selection.tools) <= {tool.name for tool in candidates}:
        raise IrisProviderError("DeepSeek 返回了候选集合之外的工具")
    return {
        "selected": selection.tools,
        "model": parsed.model,
        "input_tokens": parsed.usage.prompt_tokens,
        "output_tokens": parsed.usage.completion_tokens,
        "cache_hit_tokens": parsed.usage.prompt_cache_hit_tokens,
        "cache_miss_tokens": parsed.usage.prompt_cache_miss_tokens,
        "request": payload,
        "response": raw,
    }


def summarize_method(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总质量、重复波动和实测延迟，费用统一按输入完全未命中缓存计算。"""
    latency = sorted(row["latency_ms"] for row in rows)
    choices: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in rows:
        choices[row["id"]].add(tuple(row["selected"][:1]))
    result: dict[str, Any] = {
        **selection_metrics(rows, "selected"),
        "trials": len(rows),
        "unique_queries": len(choices),
        "top1_changed_queries": sum(len(values) > 1 for values in choices.values()),
        "median_ms": median(latency),
        "p95_ms": latency[math.ceil(len(latency) * 0.95) - 1],
        "mean_ms": sum(latency) / len(latency),
    }
    if method == "jev_choice":
        # 本轮 Choice 只选择首选，不把相对概率前三名解释为三个适用工具。
        result["top3"] = None
    for key in (
        "api_calls",
        "input_tokens",
        "output_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
    ):
        result[key] = sum(row[key] for row in rows)
    if method == "deepseek":
        models = {row["model"] for row in rows if row["api_calls"]}
        if models != {"deepseek-flash"}:
            raise IrisConfigError(
                "当前价格表只适用于实际返回的 deepseek-flash", models=sorted(models)
            )
        rates = DEEPSEEK_OFF_PEAK
        cost = (
            result["input_tokens"] * rates["cache_miss"] + result["output_tokens"] * rates["output"]
        ) / 1e6
        peak_multiplier = 2
    else:
        cost = result["input_tokens"] * jev.INPUT_USD_PER_MILLION / 1e6
        peak_multiplier = 1
    result.update(
        no_cache_usd_off_peak=cost,
        no_cache_usd_peak=cost * peak_multiplier,
        no_cache_usd_per_1000_off_peak=cost / len(rows) * 1000,
        no_cache_usd_per_1000_peak=cost * peak_multiplier / len(rows) * 1000,
    )
    return result


async def main() -> None:
    """交错执行两种候选范围、四种方法和三轮相同题目，不执行工具。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--agent-config", type=Path, default=Path("examples/chat/agent.yaml"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("tmp/tool-search-choice-comparison.json")
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats 必须大于零")
    config = init_config(env_file=args.env_file)
    model = ModelConfig.model_validate(
        yaml.safe_load(args.agent_config.read_text(encoding="utf-8"))["model"]
    )
    if model.provider != "deepseek":
        raise IrisConfigError("对照实验要求 Agent 使用 DeepSeek provider")
    deepseek = create_provider_client(model.to_model_route(), base_url=model.base_url)
    jev_key = config.provider_api_keys.get("typesafe")
    if not jev_key:
        raise IrisConfigError("缺少 IRIS_PROVIDER_API_KEYS__TYPESAFE")
    endpoint = (deepseek.base_url or "https://api.deepseek.com").rstrip("/") + "/chat/completions"
    tools = jev.load_catalog()
    catalog = [tool.definition for tool in tools]
    index = DeferredToolIndex()
    index.build(tools)
    cases = TypeAdapter(list[jev.SearchCase]).validate_json(
        jev.CASES_PATH.read_text(encoding="utf-8")
    )
    total_observations = len(cases) * args.repeats * 2 * len(METHODS)
    rng = random.Random(20260928)
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "dataset": "handwritten-bilingual-v1",
        "methods": METHODS,
        "repeats": args.repeats,
        "order_seed": 20260928,
        "agent_config": str(args.agent_config),
        "requested_deepseek_model": model.name,
        "deepseek_options": {
            key: value
            for key, value in model.to_llm_request_options().items()
            if key in {"temperature", "top_p", "max_tokens", "timeout"}
        },
        "deepseek_prompt": DEEPSEEK_PROMPT,
        "jev_model": jev.MODEL,
        "noul_threshold": jev.THRESHOLD,
        "choice_policy": "single-choice-with-none; no probability or confidence threshold",
        "choice_instructions": jev.CHOICE_INSTRUCTIONS,
        "shortlist": jev.SHORTLIST,
        "cost_assumption": "all_input_tokens_cache_miss",
        "price_date": "2026-09-28",
        "deepseek_price_model": "deepseek-flash",
        "deepseek_off_peak_per_million": DEEPSEEK_OFF_PEAK,
        "deepseek_price_source": "https://api-docs.deepseek.com/quick_start/pricing/",
        "jev_input_usd_per_million": jev.INPUT_USD_PER_MILLION,
        "jev_price_source": "https://docs.typesafe.ai/models",
        "catalog": [jev._tool_summary(tool) for tool in catalog],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx2.AsyncClient(timeout=60.0) as client:
        for repeat in range(1, args.repeats + 1):
            ordered = list(cases)
            rng.shuffle(ordered)
            for case in ordered:
                pools = ["bm25", "all"]
                rng.shuffle(pools)
                for pool in pools:
                    start = perf_counter()
                    baseline = index.search(case.query, limit=jev.SHORTLIST)
                    local_ms = (perf_counter() - start) * 1000
                    candidates = baseline if pool == "bm25" else catalog
                    common = {
                        **case.model_dump(),
                        "repeat": repeat,
                        "pool": pool,
                        "selected": [],
                        "api_calls": 0,
                        "api_ms": None,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_hit_tokens": 0,
                        "cache_miss_tokens": 0,
                        "model": None,
                        "response": None,
                    }
                    rows.append(
                        {
                            **common,
                            "method": "rules",
                            "latency_ms": local_ms,
                            "candidates": [tool.name for tool in baseline],
                            "selected": [tool.name for tool in baseline[:3]],
                        }
                    )
                    methods = list(METHODS[1:])
                    rng.shuffle(methods)
                    for method in methods:
                        row = {
                            **common,
                            "method": method,
                            "candidates": [tool.name for tool in candidates],
                            "started_at": datetime.now(UTC).isoformat(),
                            "latency_ms": local_ms if pool == "bm25" else 0.0,
                        }
                        if candidates:
                            start = perf_counter()
                            if method == "jev_noul":
                                response = await jev.query_jev(
                                    client, jev_key, case.query, candidates
                                )
                                ranked = sorted(
                                    candidates,
                                    key=lambda tool: response.answers[tool.name].noul,
                                    reverse=True,
                                )
                                row.update(
                                    selected=[
                                        tool.name
                                        for tool in ranked
                                        if response.answers[tool.name].noul >= jev.THRESHOLD
                                    ][:3],
                                    model=response.model,
                                    input_tokens=response.usage.input_tokens,
                                    output_tokens=response.usage.output_tokens,
                                    response=response.model_dump(),
                                )
                            elif method == "jev_choice":
                                row.update(
                                    await jev.evaluate_choice(
                                        client,
                                        jev_key,
                                        case.query,
                                        candidates,
                                    )
                                )
                            else:
                                row.update(
                                    await evaluate_deepseek(
                                        client,
                                        deepseek.api_key,
                                        model,
                                        case.query,
                                        candidates,
                                        endpoint=endpoint,
                                        headers=deepseek.headers,
                                    )
                                )
                            row["api_ms"] = (perf_counter() - start) * 1000
                            row["latency_ms"] += row["api_ms"]
                            row["api_calls"] = 1
                        rows.append(row)
                        args.output.write_text(
                            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                    if len(rows) % (len(METHODS) * 10) == 0:
                        print(
                            f"已完成 {len(rows)}/{total_observations} 个观测",
                            flush=True,
                        )
    report["completed_at"] = datetime.now(UTC).isoformat()
    report["summary"] = {
        pool: {
            method: summarize_method(
                method, [row for row in rows if row["pool"] == pool and row["method"] == method]
            )
            for method in METHODS
        }
        for pool in ("bm25", "all")
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"结果已保存：{args.output.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
