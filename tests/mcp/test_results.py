"""MCP 模型正文和完整 artifact 的集成契约。"""

import json
from pathlib import Path
from typing import Any

import pytest
from mcp import types

from iris.exceptions import IrisToolExecutionError
from iris.mcp.models import MCPResolvedServer
from iris.message import TextBlock, ToolUseBlock
from iris.tools import (
    BaseTool,
    ToolArtifact,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolRegistry,
    ToolResult,
)
from iris.tools.artifacts import ToolArtifactStore

from .fixtures.tools import make_tool


@pytest.mark.asyncio
@pytest.mark.parametrize("rich", [False, True])
async def test_middleware_expansion_preserves_original_json_when_present(
    stdio_config: MCPResolvedServer,
    tmp_path: Path,
    rich: bool,
) -> None:
    class Expand(ToolMiddleware):
        async def after_call(
            self, tool: BaseTool, result: ToolResult, context: ToolExecutionContext
        ) -> ToolResult:
            """模拟正常的结果扩展 hook。"""
            return result.model_copy(update={"content": [TextBlock(text="expanded" * 2000)]})

    source = types.CallToolResult(content=[types.TextContent(text="original")])
    if rich:
        source.structured_content = {"value": 9}
    tool, _ = make_tool(stdio_config, result=source)
    tool.definition.max_result_chars = 1000
    registry = ToolRegistry()
    registry.register(tool)
    result = await ToolExecutor(registry, middleware=[Expand()]).execute_one(
        ToolUseBlock(id="call", name=tool.name, input={}),
        ToolExecutionContext(workspace_root=tmp_path, session_id="session"),
    )
    assert len(result.model_content) <= 1000
    assert result.artifact.path.suffix == (".json" if rich else ".txt")
    text = result.artifact.path.read_text(encoding="utf-8")
    assert (
        (json.loads(text)["structuredContent"] == {"value": 9})
        if rich
        else text == "expanded" * 2000
    )
    assert str(result.artifact.path) in result.model_content


@pytest.mark.asyncio
async def test_rich_content_and_extensions_survive_in_json(
    stdio_config: MCPResolvedServer, tmp_path: Path
) -> None:
    source = types.CallToolResult(
        content=[
            types.TextContent(text="first"),
            types.ResourceLink(
                uri="https://example.test/item", name="item", mime_type="text/plain"
            ),
            types.EmbeddedResource(
                resource=types.TextResourceContents(uri="file:///note", text="embedded")
            ),
            types.ImageContent(data="YWJj", mime_type="image/png"),
            types.AudioContent(data="YWJj", mime_type="audio/wav"),
            types.TextContent(text="last"),
        ],
        structured_content={"answer": 42},
        _meta={"vendor": "preserved"},
    )
    tool, _ = make_tool(stdio_config, result=source)
    result = await tool.arun(
        {}, ToolExecutionContext(workspace_root=tmp_path, call_id="call/1", session_id="one")
    )
    assert result.artifact.mime_type == "application/json"
    payload = json.loads(result.artifact.path.read_text(encoding="utf-8"))
    assert payload["content"][3]["data"] == "YWJj" and payload["_meta"]["vendor"] == "preserved"
    assert (
        "embedded" in result.model_content and "https://example.test/item" in result.model_content
    )
    assert "YWJj" not in result.model_content
    assert result.model_content.index("first") < result.model_content.index("last")
    second = await tool.arun(
        {}, ToolExecutionContext(workspace_root=tmp_path, call_id="call/1", session_id="two")
    )
    assert result.artifact.path != second.artifact.path and result.artifact.path.is_file()


@pytest.mark.asyncio
async def test_long_error_is_bounded_with_visible_artifact(
    stdio_config: MCPResolvedServer, tmp_path: Path
) -> None:
    source = types.CallToolResult(
        content=[types.TextContent(text="failure " * 12000)], is_error=True
    )
    tool, _ = make_tool(stdio_config, result=source)
    tool.definition.max_result_chars = 1000
    registry = ToolRegistry()
    registry.register(tool)
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="error", name=tool.name, input={}),
        ToolExecutionContext(workspace_root=tmp_path, session_id="session"),
    )
    assert result.error.code == "MCP_TOOL_ERROR"
    assert len(result.model_content) <= 1000 and str(result.artifact.path) in result.error.message
    assert json.loads(result.artifact.path.read_text())["content"][0]["text"] == "failure " * 12000


@pytest.mark.asyncio
async def test_json_write_failure_is_known_local_error_without_replay(
    stdio_config: MCPResolvedServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write(self: ToolArtifactStore, *args: Any, **kwargs: Any) -> ToolArtifact:
        """模拟本地写入失败。"""
        raise IrisToolExecutionError("ARTIFACT_ERROR: failed")

    monkeypatch.setattr(ToolArtifactStore, "persist_json", fail_write)
    tool, connection = make_tool(
        stdio_config, result=types.CallToolResult(content=[], structured_content={"x": 1})
    )
    result = await tool.arun({}, ToolExecutionContext(workspace_root=tmp_path, call_id="call"))
    assert result.error.code == "ARTIFACT_ERROR" and result.artifact is None
    assert len(connection.calls) == 1
