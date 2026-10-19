"""Web 示例的离线命令与工具执行契约。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest

from examples.web import tools as web_example
from iris.config import init_config, reset


@pytest.mark.parametrize("failed", [False, True])
def test_fetch_command_reports_partial_and_total_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    failed: bool,
) -> None:
    """真实 executor 将批量失败呈现给命令行，并给出对应退出码。"""
    requests: list[dict[str, Any]] = []
    original_client = httpx2.AsyncClient

    def respond(request: httpx2.Request) -> httpx2.Response:
        import json

        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "results": []
                if failed
                else [{"url": "https://example.com/", "raw_content": "正文"}],
                "failed_results": [{"url": "https://example.com/missing", "error": "404"}],
            },
        )

    def client(**kwargs: Any) -> httpx2.AsyncClient:
        return original_client(transport=httpx2.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx2, "AsyncClient", client)
    reset()
    init_config(provider_api_keys={"tavily": "test-key"})
    try:
        code = web_example.main(
            [
                "--workspace",
                str(tmp_path),
                "fetch",
                "https://example.com/",
                "https://example.com/missing",
                "--query",
                "正文",
            ]
        )
    finally:
        reset()
    output = capsys.readouterr().out
    assert code == int(failed)
    assert "https://example.com/missing" in output
    assert "404" in output
    assert ("EXECUTION_ERROR" in output) is failed
    assert requests == [
        {
            "urls": ["https://example.com/", "https://example.com/missing"],
            "query": "正文",
            "extract_depth": "basic",
            "format": "markdown",
        }
    ]


@pytest.mark.asyncio
async def test_read_command_resumes_artifact_without_api_key(tmp_path: Path) -> None:
    """续读使用 read_file 的 offset/column，且不装配 Web 工具。"""
    artifact = tmp_path / "result.txt"
    artifact.write_text("第一行\n第二行正文", encoding="utf-8")
    result = await web_example.execute_tool(
        "read_file",
        {"file_path": str(artifact), "offset": 1, "column": 3},
        workspace=tmp_path,
    )
    assert not result.is_error
    assert "正文" in result.model_content
    assert "第一行" not in result.model_content


def test_module_command_prints_unicode_when_pipe_defaults_to_gbk(tmp_path: Path) -> None:
    """复现 Windows 管道的 GBK 编码无法输出真实网页字符的问题。"""
    artifact = tmp_path / "result.txt"
    artifact.write_text("Python 文档 ¶ 😀", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.web.tools",
            "--workspace",
            str(tmp_path),
            "read",
            artifact.name,
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONIOENCODING": "gbk"},
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Python 文档 ¶ 😀" in completed.stdout


@pytest.mark.parametrize("command", ["search", "fetch"])
def test_deepseek_command_uses_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    command: str,
) -> None:
    """示例 CLI 透传 DeepSeek 的查询或单页 URL，不夹带 Tavily 参数。"""
    requests: list[httpx2.Request] = []
    original_client = httpx2.AsyncClient

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={"results": []} if command == "search" else {"title": "Python", "content": "正文"},
        )

    def client(**kwargs: Any) -> httpx2.AsyncClient:
        return original_client(transport=httpx2.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx2, "AsyncClient", client)
    reset()
    init_config(provider_api_keys={"deepseek": "test-key"})
    try:
        code = web_example.main(
            [
                "--provider",
                "deepseek",
                "--workspace",
                str(tmp_path),
                command,
                "Python site:python.org" if command == "search" else "https://www.python.org/",
            ]
        )
    finally:
        reset()
    assert code == 0
    assert len(requests) == 1
    assert str(requests[0].url) == f"https://api.deepseek.com/v1/tools/{command}"
    assert json.loads(requests[0].content) == (
        {"query": "Python site:python.org", "count": 10, "need_content": False}
        if command == "search"
        else {"url": "https://www.python.org/"}
    )
    assert ("Results: 0" if command == "search" else "正文") in capsys.readouterr().out
