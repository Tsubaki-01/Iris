"""验证模型对照实验的请求、外部结果和无缓存费用口径。"""

import json
from typing import Any

import httpx2
import pytest

from evals.tool_search_comparison import evaluate_deepseek, summarize_method
from iris.agents.config.base import ModelConfig
from iris.exceptions import IrisProviderError
from iris.tools import ToolDefinition


def _raw(content: str) -> dict[str, Any]:
    return {
        "model": "deepseek-flash",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 20,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200,
        },
    }


@pytest.mark.asyncio
async def test_deepseek_keeps_configured_model_and_raw_cache_usage() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-chat"
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 1024
        assert payload["response_format"] == {"type": "json_object"}
        assert "expected" not in request.content.decode()
        assert json.loads(payload["messages"][1]["content"])["query"] == "读取说明"
        return httpx2.Response(200, json=_raw('{"tools":["read_file"]}'))

    tool = ToolDefinition(name="read_file", description="读取文件", input_schema={"type": "object"})
    model = ModelConfig(provider="deepseek", name="deepseek-chat", temperature=0.2, max_tokens=1024)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        result = await evaluate_deepseek(client, "test-key", model, "读取说明", [tool])
    assert result["selected"] == ["read_file"]
    assert result["model"] == "deepseek-flash"
    assert result["cache_hit_tokens"] == 800
    assert result["input_tokens"] == 1000
    assert result["response"]["usage"]["prompt_cache_miss_tokens"] == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ['{"tools":["unknown"]}', '{"tools":"read_file"}'])
async def test_invalid_model_choice_is_not_silently_filtered(content: str) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_raw(content))

    tool = ToolDefinition(name="read_file", description="读取文件", input_schema={"type": "object"})
    model = ModelConfig(provider="deepseek", name="deepseek-chat")
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        with pytest.raises(IrisProviderError):
            await evaluate_deepseek(client, "test-key", model, "读取说明", [tool])


@pytest.mark.parametrize("cache_hit_tokens", [0, 800, 1000])
def test_cost_assumes_no_cache_regardless_of_observed_hits(cache_hit_tokens: int) -> None:
    rows = [
        {
            "id": "read",
            "repeat": 1,
            "expected": ["read_file"],
            "selected": ["read_file"],
            "candidates": ["read_file"],
            "latency_ms": 500,
            "api_ms": 499,
            "api_calls": 1,
            "model": "deepseek-flash",
            "input_tokens": 1000,
            "output_tokens": 20,
            "cache_hit_tokens": cache_hit_tokens,
            "cache_miss_tokens": 1000 - cache_hit_tokens,
        }
    ]
    result = summarize_method("deepseek", rows)
    assert result["top1"] == 1
    assert result["cache_hit_tokens"] == cache_hit_tokens
    assert result["cache_miss_tokens"] == 1000 - cache_hit_tokens
    assert result["no_cache_usd_off_peak"] == pytest.approx(0.000162)
    assert result["no_cache_usd_peak"] == pytest.approx(0.000324)
    assert result["no_cache_usd_per_1000_off_peak"] == pytest.approx(0.162)
    assert result["no_cache_usd_per_1000_peak"] == pytest.approx(0.324)


def test_repeats_report_unique_queries_and_selection_changes() -> None:
    common = {
        "id": "read",
        "expected": ["a"],
        "candidates": ["a", "b"],
        "latency_ms": 1,
        "api_ms": None,
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 0,
    }
    result = summarize_method(
        "rules",
        [
            dict(common, repeat=1, selected=["a"]),
            dict(common, repeat=2, selected=["b"]),
        ],
    )
    assert result["unique_queries"] == 1
    assert result["trials"] == 2
    assert result["top1"] == 0.5
    assert result["top1_changed_queries"] == 1


def test_choice_reports_single_selection_metrics_and_actual_input_cost() -> None:
    row = {
        "id": "read",
        "expected": ["read_file"],
        "selected": ["read_file"],
        "candidates": ["read_file"],
        "latency_ms": 300,
        "api_calls": 1,
        "input_tokens": 500,
        "output_tokens": 80,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 0,
    }
    result = summarize_method("jev_choice", [row])
    assert result["top1"] == 1
    assert result["top3"] is None
    assert result["no_cache_usd_off_peak"] == pytest.approx(500 * 0.042 / 1e6)
    assert result["no_cache_usd_per_1000_off_peak"] == pytest.approx(0.021)
