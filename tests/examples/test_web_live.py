"""显式开启的真实 Tavily 与 DeepSeek 验证；不替换 provider 或 HTTP transport。"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

import pytest

from examples.web.agent import CONFIG_PATH, DEFAULT_PROMPT, run_agent
from examples.web.tools import execute_tool
from iris.config import init_config, reset
from iris.lifecycle import RunPhase, RunStopReason, ToolCallPhase
from iris.store import SQLiteStore
from iris.tools import ToolResult

pytestmark = [pytest.mark.live_web, pytest.mark.usefixtures("live_config"), pytest.mark.asyncio]

PAGES = [
    "https://docs.python.org/3/library/asyncio-task.html",
    "https://peps.python.org/pep-0654/",
]
MISSING = "https://docs.python.org/3/iris-web-example-missing.html"
QUERY = "How does TaskGroup cancel sibling tasks and raise ExceptionGroup?"
LiveCall = Callable[[str, dict[str, Any]], Awaitable[tuple[ToolResult, str]]]


@pytest.fixture
def live_config(request: pytest.FixtureRequest) -> Iterator[None]:
    """只有显式启用时才加载凭据；启用后配置错误应直接失败。"""
    if not request.config.getoption("--run-live-web"):
        pytest.skip("使用 --run-live-web 显式开启真实 API 调用")
    reset()
    init_config(env_file=request.config.getoption("--web-env-file"))
    try:
        yield
    finally:
        reset()


@pytest.fixture
def live_call(tmp_path: Path) -> LiveCall:
    """保存调用参数和标准结果，长正文保存为 Markdown，均不包含认证配置。"""
    sequence = 0

    async def call(name: str, arguments: dict[str, Any]) -> tuple[ToolResult, str]:
        nonlocal sequence
        sequence += 1
        result = await execute_tool(name, arguments, workspace=tmp_path)
        prefix = tmp_path / f"{sequence}-{name}"
        prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "arguments": arguments,
                    "result": result.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        content = (
            result.artifact.path.read_text(encoding="utf-8")
            if result.artifact is not None
            else result.model_content
        )
        prefix.with_suffix(".md").write_text(content, encoding="utf-8")
        print(f"{name}: error={result.is_error}, chars={len(content)}, evidence={prefix}.json")
        return result, content

    return call


def _urls(content: str) -> list[str]:
    """提取模型实际收到的 Search 来源 URL。"""
    return re.findall(r"^URL: (.+)$", content, re.MULTILINE)


def _python_document(url: str) -> str:
    """本例只比较 Python 文档；highlight 和锚点不改变文档身份。"""
    parts = urlsplit(url)
    query = urlencode([(key, value) for key, value in parse_qsl(parts.query) if key != "highlight"])
    return parts._replace(query=query, fragment="").geturl()


async def test_search_defaults(live_call: LiveCall) -> None:
    """省略可选参数，保留可继续读取的标题、URL 与片段。"""
    result, content = await live_call("web_search", {"query": "Python asyncio TaskGroup"})
    assert not result.is_error, content
    assert 0 < len(_urls(content)) <= 10
    assert "## 1. " in content and "TaskGroup" in content


async def test_search_domain_filters(live_call: LiveCall) -> None:
    """验证实际返回域名同时满足包含与排除条件。"""
    result, content = await live_call(
        "web_search",
        {
            "query": "Python asyncio TaskGroup ExceptionGroup",
            "max_results": 3,
            "include_domains": ["docs.python.org", "peps.python.org"],
            "exclude_domains": ["peps.python.org"],
        },
    )
    urls = _urls(content)
    assert not result.is_error, content
    assert 0 < len(urls) <= 3
    assert all(urlsplit(url).hostname == "docs.python.org" for url in urls)


async def test_search_exclude_only(live_call: LiveCall) -> None:
    """排除域名参数可独立使用。"""
    result, content = await live_call(
        "web_search",
        {
            "query": "Python asyncio TaskGroup",
            "max_results": 3,
            "exclude_domains": ["docs.python.org"],
        },
    )
    urls = _urls(content)
    assert not result.is_error and 0 < len(urls) <= 3, content
    assert all(urlsplit(url).hostname != "docs.python.org" for url in urls)


@pytest.mark.parametrize("time_range", ["day", "week", "month", "year"])
async def test_search_time_ranges(live_call: LiveCall, time_range: str) -> None:
    """服务接受四种时间范围；输出不含日期，不能独立证明严格发布日期。"""
    result, content = await live_call(
        "web_search",
        {
            "query": "Python release announcement",
            "max_results": 3,
            "time_range": time_range,
        },
    )
    assert not result.is_error, content
    assert "Results:" in content and len(_urls(content)) <= 3


async def test_search_empty_results(live_call: LiveCall) -> None:
    """搜索不存在的限定域名时，零结果仍为成功。"""
    result, content = await live_call(
        "web_search",
        {
            "query": "iris web example",
            "include_domains": ["iris-web-example-missing.python.org"],
        },
    )
    assert not result.is_error and "Results: 0" in content, content


async def test_search_http_error(live_call: LiveCall) -> None:
    """服务拒绝无效顶级域名时，保留真实 HTTP 错误与原因。"""
    result, content = await live_call(
        "web_search",
        {
            "query": "iris web example",
            "include_domains": ["iris-web-example.invalid"],
        },
    )
    assert result.is_error and result.error is not None
    assert result.error.code == "EXECUTION_ERROR"
    assert "HTTP 400" in content and "invalid" in content


async def test_full_content_and_artifact_read_to_end(live_call: LiveCall) -> None:
    """真实长正文触发默认 artifact，并通过文件工具逐页还原全文。"""
    result, content = await live_call("web_fetch", {"urls": PAGES})
    assert not result.is_error, content
    assert "Mode: full_content" in content and "Succeeded: 2" in content
    assert "Failed: 0" in content and all(url in content for url in PAGES)
    assert result.artifact is not None, "这些长页面应触发默认 50,000 字符阈值"
    offset, column = 0, 0
    parts: list[str] = []
    while True:
        page, text = await live_call(
            "read_file",
            {
                "file_path": str(result.artifact.path),
                "offset": offset,
                "column": column,
            },
        )
        assert not page.is_error, text
        body, footer = text.rsplit("\n\n[read_file: ", 1)
        parts.append(body)
        cursor = re.search(r"next_offset=(\d+), next_column=(\d+); has_more=(true|false)", footer)
        assert cursor is not None, footer
        if cursor[3] == "false":
            break
        next_cursor = int(cursor[1]), int(cursor[2])
        assert next_cursor > (offset, column), "续读游标必须推进"
        offset, column = next_cursor
    assert "".join(parts) == content


async def test_excerpts_follow_query(live_call: LiveCall) -> None:
    """同一页面切换问题时摘录变化，且包含回答问题所需概念。"""
    result, task_group = await live_call("web_fetch", {"urls": PAGES, "query": QUERY})
    assert not result.is_error, task_group
    assert "Mode: excerpts" in task_group and "Succeeded: 2" in task_group
    body = task_group.split("## Content:", 1)[1]
    assert "ExceptionGroup" in body and "cancel" in body.lower()
    result, to_thread = await live_call(
        "web_fetch",
        {
            "urls": [PAGES[0]],
            "query": "How does asyncio.to_thread run blocking IO?",
        },
    )
    assert not result.is_error, to_thread
    other_body = to_thread.split("## Content:", 1)[1]
    assert "to_thread" in other_body and "blocking" in other_body.lower()
    assert other_body != body.split("## Content:")[0]


@pytest.mark.parametrize("query", [None, QUERY], ids=["full", "excerpts"])
@pytest.mark.parametrize("partial", [True, False], ids=["partial", "all-failed"])
async def test_fetch_failures(live_call: LiveCall, query: str | None, partial: bool) -> None:
    """两种读取模式均保留失败 URL；部分成功继续交付正文，全部失败返回工具错误。"""
    urls = ["https://docs.python.org/3/library/asyncio.html", MISSING] if partial else [MISSING]
    result, content = await live_call("web_fetch", {"urls": urls, "query": query})
    assert result.is_error is (not partial), content
    assert MISSING in content and "Failed: 1" in content and "Reason:" in content
    if partial:
        assert "Succeeded: 1" in content and f"## Content: {urls[0]}" in content
        assert content.index("## Failed URLs") < content.index("## Content:")
    else:
        assert result.error is not None and result.error.code == "EXECUTION_ERROR"
        assert "Succeeded: 0" in content


async def test_real_model_search_fetch_and_sqlite(tmp_path: Path) -> None:
    """真实模型根据搜索结果调用 Extract，给出有来源的答案，SQLite 重读闭环。"""
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    result = await run_agent(config_path=config_path, prompt=DEFAULT_PROMPT, session_id="live-web")
    (tmp_path / "run-result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    store = SQLiteStore(tmp_path / ".iris" / "web.db")
    calls = store.list_tool_calls(result.run.run_id)
    (tmp_path / "tool-calls.json").write_text(
        json.dumps(
            [call.model_dump(mode="json") for call in calls],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Agent evidence={tmp_path}, model_steps={result.run.usage.model_steps_committed}")
    assert result.run.phase is RunPhase.TERMINAL
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert result.error is None
    assert all(call.phase is ToolCallPhase.COMMITTED and call.result is not None for call in calls)
    searches = [call for call in calls if call.tool_name == "web_search"]
    fetches = [call for call in calls if call.tool_name == "web_fetch"]
    assert searches and fetches, "模型必须实际调用搜索与读取，不能只生成回答"
    assert all(not call.result.is_error for call in searches + fetches)
    sources = {url for call in searches for url in _urls(call.result.model_content)}
    fetched = {url for call in fetches for url in call.arguments["urls"]}
    assert {_python_document(url) for url in fetched} <= {
        _python_document(url) for url in sources
    }, "模型应读取真实搜索返回的文档；允许省略文档高亮参数和锚点"
    assert result.assistant_message is not None
    answer = result.assistant_message.text
    assert "ExceptionGroup" in answer and any(url in answer for url in fetched)
    assert store.load_result(result.run.run_id) == result
    history = store.load_session("live-web")
    results = [block for message in history.messages for block in message.tool_results]
    assert {block.tool_use_id for block in results} == {call.tool_call_id for call in calls}
