"""Tavily Web 工具的请求、正文和失败契约。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx2
import pytest

from iris.exceptions import IrisToolExecutionError, IrisToolValidationError
from iris.message import ToolUseBlock
from iris.tools import (
    BaseTool,
    PermissionDecision,
    PermissionEffect,
    PermissionPolicy,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
)
from iris.tools.builtin import _tavily
from iris.tools.builtin.web import WebFetchTool, WebSearchTool


def _mock_http(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx2.Request], httpx2.Response | Awaitable[httpx2.Response]],
) -> list[httpx2.AsyncClient]:
    """保留真实客户端行为，只用 transport 替换外部服务。"""
    original_client = httpx2.AsyncClient
    clients: list[httpx2.AsyncClient] = []

    def create_client(**kwargs: Any) -> httpx2.AsyncClient:
        client = original_client(transport=httpx2.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(_tavily.httpx2, "AsyncClient", create_client)
    return clients


@pytest.mark.asyncio
async def test_search_maps_filters_and_preserves_source_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """服务片段和 URL 进入模型正文，过滤参数明确限制来源。"""
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "results": [
                    {"title": "Asyncio", "url": "https://docs.python.org/a", "content": "片段甲"},
                    {"title": "PEP", "url": "https://peps.python.org/b", "content": "片段乙"},
                ],
                "request_id": "service-id",
            },
        )

    clients = _mock_http(monkeypatch, respond)
    tool = WebSearchTool(api_key="test-key")
    result = await tool.arun(
        tool.validate_input(
            {
                "query": "asyncio",
                "max_results": 2,
                "time_range": "month",
                "include_domains": ["python.org"],
                "exclude_domains": ["old.python.org"],
            }
        ),
        ToolExecutionContext(workspace_root=".", call_id="search-call"),
    )

    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "https://api.tavily.com/search"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content) == {
        "query": "asyncio",
        "max_results": 2,
        "time_range": "month",
        "include_domains": ["python.org"],
        "exclude_domains": ["old.python.org"],
        "include_domains_mode": "filter",
        "search_depth": "basic",
        "topic": "general",
        "include_answer": False,
        "include_raw_content": False,
        "auto_parameters": False,
    }
    assert result.model_content == (
        "# Web Search\n\nQuery: asyncio\nResults: 2\n\n"
        "## 1. Asyncio\nURL: https://docs.python.org/a\n\n片段甲\n\n"
        "## 2. PEP\nURL: https://peps.python.org/b\n\n片段乙"
    )
    assert result.data == {}
    assert result.tool_use_id == "search-call"
    assert clients[0].is_closed
    assert clients[0].timeout == httpx2.Timeout(30.0)


@pytest.mark.asyncio
async def test_search_defaults_and_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """未指定过滤条件时不添加筛选；零结果是正常工具结果。"""
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"results": []})

    _mock_http(monkeypatch, respond)
    tool = WebSearchTool(api_key="test-key")
    result = await tool.arun(
        tool.validate_input({"query": "nothing found"}),
        ToolExecutionContext(workspace_root="."),
    )

    payload = json.loads(requests[0].content)
    assert payload["max_results"] == 10
    assert not {"time_range", "include_domains", "exclude_domains", "include_domains_mode"} & (
        payload.keys()
    )
    assert "Results: 0" in result.model_content
    assert not result.is_error


@pytest.mark.parametrize(
    "params",
    [
        {"query": " "},
        {"query": "x", "max_results": 0},
        {"query": "x", "max_results": 21},
        {"query": "x", "time_range": "hour"},
        {"query": "x", "include_domains": ["example.com"] * 301},
        {"query": "x", "exclude_domains": ["example.com"] * 151},
        {"query": "x", "search_depth": "advanced"},
    ],
)
def test_search_validates_public_parameters(params: dict[str, Any]) -> None:
    """输入边界拥有参数约束，固定深度不能被工具参数覆盖。"""
    with pytest.raises(IrisToolValidationError):
        WebSearchTool(api_key="test-key").validate_input(params)


@pytest.mark.parametrize(
    "params",
    [
        {"urls": []},
        {"urls": ["https://example.com/page"] * 21},
        {"urls": ["file:///tmp/page"]},
        {"urls": ["not a URL"]},
        {"urls": ["https://example.com/page"], "query": " "},
    ],
)
def test_fetch_validates_public_parameters(params: dict[str, Any]) -> None:
    """Fetch 只接受 1 至 20 个 HTTP(S) URL 和可选非空 query。"""
    with pytest.raises(IrisToolValidationError):
        WebFetchTool(api_key="test-key").validate_input(params)


def test_fetch_arguments_support_durable_and_human_gate_projection() -> None:
    """已校验的 URL 在 executor 参数投影中可用于 durable fingerprint 与 HITL。"""

    class RequireHuman(PermissionPolicy):
        """要求 Web 工具进入正常人工授权流程的宿主策略。"""

        def check(
            self, tool: BaseTool, params: dict[str, Any], context: ToolExecutionContext
        ) -> PermissionDecision:
            """保留调用参数并要求人工确认。"""
            return PermissionDecision(effect=PermissionEffect.REQUIRE_HUMAN, reason="host policy")

    registry = ToolRegistry()
    registry.register(WebFetchTool(api_key="test-key"))
    prepared = (
        ToolExecutor(registry, permission_policy=RequireHuman())
        .prepare_many(
            [
                ToolUseBlock(
                    id="fetch-call", name="web_fetch", input={"urls": ["https://example.com/page"]}
                )
            ],
            ToolExecutionContext(workspace_root=".", session_id="s", metadata={"run_id": "r"}),
        )
        .calls[0]
    )

    assert prepared.preflight_result is None
    assert prepared.human_request is not None
    assert prepared.arguments == {"urls": ["https://example.com/page"], "query": None}
    assert prepared.human_request.tool_call.arguments == prepared.arguments


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [None, "错误处理"])
async def test_fetch_preserves_all_content_and_partial_failures(
    monkeypatch: pytest.MonkeyPatch, query: str | None
) -> None:
    """按响应 URL 标识正文，不把返回顺序或 title 当作 API 保证。"""
    requests: list[httpx2.Request] = []
    urls = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "results": [
                    {"url": urls[2], "raw_content": "# 原始标题\n\n正文丙"},
                    {"url": urls[0], "raw_content": "正文甲"},
                ],
                "failed_results": [{"url": urls[1], "error": "页面不可用"}],
            },
        )

    _mock_http(monkeypatch, respond)
    tool = WebFetchTool(api_key="test-key")
    params: dict[str, Any] = {"urls": urls}
    if query is not None:
        params["query"] = query
    result = await tool.arun(tool.validate_input(params), ToolExecutionContext(workspace_root="."))

    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.tavily.com/extract"
    payload = json.loads(requests[0].content)
    expected: dict[str, Any] = {"urls": urls, "extract_depth": "basic", "format": "markdown"}
    if query is not None:
        expected["query"] = query
    assert payload == expected
    assert not result.is_error
    assert result.data == {}
    mode = "full_content" if query is None else "excerpts"
    assert f"Mode: {mode}" in result.model_content
    assert ("Query:" in result.model_content) is (query is not None)
    assert "Succeeded: 2\nFailed: 1" in result.model_content
    assert f"- {urls[1]}\n  Reason: 页面不可用" in result.model_content
    assert f"## Content: {urls[2]}\n\n# 原始标题\n\n正文丙" in result.model_content
    assert f"## Content: {urls[0]}\n\n正文甲" in result.model_content
    assert result.model_content.index("## Failed URLs") < result.model_content.index("## Content:")


@pytest.mark.asyncio
async def test_fetch_all_failed_is_an_error_with_each_url_and_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 200 的全部失败也必须让 executor 投影为工具错误。"""
    _mock_http(
        monkeypatch,
        lambda request: httpx2.Response(
            200,
            json={
                "results": [],
                "failed_results": [
                    {"url": "https://example.com/a", "error": "timeout"},
                    {"url": "https://example.com/b", "error": "not found"},
                ],
            },
        ),
    )
    tool = WebFetchTool(api_key="test-key")
    with pytest.raises(IrisToolExecutionError) as raised:
        await tool.arun(
            tool.validate_input({"urls": ["https://example.com/a", "https://example.com/b"]}),
            ToolExecutionContext(workspace_root="."),
        )
    message = str(raised.value)
    assert "https://example.com/a" in message and "timeout" in message
    assert "https://example.com/b" in message and "not found" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 432, 500])
async def test_http_failures_preserve_service_reason_and_do_not_retry(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """HTTP 错误保留状态及服务消息；一次工具调用只发一次请求。"""
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(status, json={"detail": "service failure reason"})

    clients = _mock_http(monkeypatch, respond)
    tool = WebSearchTool(api_key="test-key")
    with pytest.raises(IrisToolExecutionError) as raised:
        await tool.arun(
            tool.validate_input({"query": "x"}), ToolExecutionContext(workspace_root=".")
        )
    assert str(status) in str(raised.value)
    assert "service failure reason" in str(raised.value)
    assert len(requests) == 1
    assert clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [b"not JSON", b'{"error":"failed"}', b'{"results":[{}]}'])
async def test_search_rejects_unparseable_responses(
    monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    """外部响应缺少消费字段时不能伪装成搜索无结果。"""
    _mock_http(monkeypatch, lambda request: httpx2.Response(200, content=content))
    tool = WebSearchTool(api_key="test-key")
    with pytest.raises(IrisToolExecutionError, match="响应"):
        await tool.arun(
            tool.validate_input({"query": "x"}), ToolExecutionContext(workspace_root=".")
        )


@pytest.mark.asyncio
async def test_fetch_rejects_missing_raw_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺正文的成功项是解析失败，不能生成成功空正文。"""
    _mock_http(
        monkeypatch,
        lambda request: httpx2.Response(
            200,
            json={"results": [{"url": "https://example.com/a"}], "failed_results": []},
        ),
    )
    tool = WebFetchTool(api_key="test-key")
    with pytest.raises(IrisToolExecutionError, match="响应"):
        await tool.arun(
            tool.validate_input({"urls": ["https://example.com/a"]}),
            ToolExecutionContext(workspace_root="."),
        )


@pytest.mark.asyncio
async def test_network_failure_is_distinct_from_response_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """网络超时转换为工具请求失败并关闭客户端。"""

    def timeout(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("read timed out", request=request)

    clients = _mock_http(monkeypatch, timeout)
    tool = WebSearchTool(api_key="test-key")
    with pytest.raises(IrisToolExecutionError, match="请求失败"):
        await tool.arun(
            tool.validate_input({"query": "x"}), ToolExecutionContext(workspace_root=".")
        )
    assert clients[0].is_closed


@pytest.mark.asyncio
async def test_cancellation_propagates_and_closes_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """取消正在等待网络的工具时不吞掉取消信号。"""
    started = asyncio.Event()

    async def wait_for_cancellation(request: httpx2.Request) -> httpx2.Response:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("请求应被取消")

    clients = _mock_http(monkeypatch, wait_for_cancellation)
    tool = WebFetchTool(api_key="test-key")
    task = asyncio.create_task(
        tool.arun(
            tool.validate_input({"urls": ["https://example.com/a"]}),
            ToolExecutionContext(workspace_root="."),
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert clients[0].is_closed
