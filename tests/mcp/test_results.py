"""MCP 模型正文和完整 artifact 的集成契约。"""

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from mcp import types
from PIL import Image

from iris.exceptions import IrisImageError, IrisToolExecutionError
from iris.mcp.models import MCPResolvedServer
from iris.message import ImageBlock, TextBlock, ToolUseBlock
from iris.runtime.streaming import RuntimeStreamEvent
from iris.streaming.projection import project_live_fact
from iris.tools import (
    ToolArtifact,
    ToolCall,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolNext,
    ToolRegistry,
    ToolResult,
)
from iris.tools._paths import safe_path_segment
from iris.tools.artifacts import ToolArtifactStore

from .fixtures.tools import AllowTools, make_tool


def _image_content(color: str = "red") -> types.ImageContent:
    """真实 PNG 数据配上错误声明 MIME，实际格式由图片导入边界判断。"""
    output = BytesIO()
    with Image.new("RGB", (12, 8), color) as image:
        image.save(output, format="PNG")
    return types.ImageContent(
        data=base64.b64encode(output.getvalue()).decode("ascii"), mime_type="image/jpeg"
    )


@pytest.mark.asyncio
async def test_middleware_receives_full_mcp_text_before_final_limit(
    stdio_config: MCPResolvedServer, tmp_path: Path
) -> None:
    """MCP 的完整正文先交给 hook，最终预览遵守同一个工具配置。"""
    text = "source" * 1000
    observed: list[str] = []

    class Observe(ToolMiddleware):
        """观察 executor 最终裁剪之前的文本。"""

        async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
            """保留原结果并记录完整正文。"""
            result = await call_next()
            observed.append(result.model_content)
            return result

    tool, _ = make_tool(
        stdio_config, result=types.CallToolResult(content=[types.TextContent(text=text)])
    )
    tool.definition.max_result_chars = 1000
    tool.definition.preview_chars = 12
    registry = ToolRegistry()
    registry.register(tool)
    result = await ToolExecutor(registry, middleware=[Observe()]).execute_one(
        ToolUseBlock(id="full-mcp", name=tool.name, input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert observed[0].startswith(text)
    assert result.model_content.startswith(text[:12] + "\n\n[")
    assert len(result.model_content) <= 1000
    assert result.artifact is not None
    assert (
        json.loads(result.artifact.path.read_text(encoding="utf-8"))["content"][0]["text"] == text
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rich", [False, True])
async def test_middleware_expansion_preserves_original_json_when_present(
    stdio_config: MCPResolvedServer,
    tmp_path: Path,
    rich: bool,
) -> None:
    class Expand(ToolMiddleware):
        async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
            """模拟正常的结果扩展 hook。"""
            result = await call_next()
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
    assert result.artifact.text_path.read_text(encoding="utf-8") == "expanded" * 2000


@pytest.mark.asyncio
async def test_rich_content_and_extensions_survive_in_json(
    stdio_config: MCPResolvedServer, tmp_path: Path
) -> None:
    image = _image_content()
    source = types.CallToolResult(
        content=[
            types.TextContent(text="first"),
            types.ResourceLink(
                uri="https://example.test/item", name="item", mime_type="text/plain"
            ),
            types.EmbeddedResource(
                resource=types.TextResourceContents(uri="file:///note", text="embedded")
            ),
            image,
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
    assert payload["content"][3]["data"] == image.data
    assert payload["_meta"]["vendor"] == "preserved"
    assert (
        "embedded" in result.model_content and "https://example.test/item" in result.model_content
    )
    assert "YWJj" not in result.model_content
    assert image.data not in result.model_content
    images = [part for part in result.content if isinstance(part, ImageBlock)]
    assert len(images) == 1
    assert images[0].model.mime_type == "image/png"
    assert images[0].original.path.read_bytes() == base64.b64decode(image.data)
    assert result.model_content.index("first") < result.model_content.index("last")
    second = await tool.arun(
        {}, ToolExecutionContext(workspace_root=tmp_path, call_id="call/1", session_id="two")
    )
    assert result.artifact.path != second.artifact.path and result.artifact.path.is_file()


@pytest.mark.asyncio
async def test_executor_keeps_mcp_mixed_images_and_raw_json(
    stdio_config: MCPResolvedServer, tmp_path: Path
) -> None:
    """实际 executor 最终归一化与 to_msg 不把图像退化为 artifact 预览。"""
    first, second = _image_content("red"), _image_content("blue")
    source = types.CallToolResult(
        content=[
            types.TextContent(text="first"),
            first,
            types.TextContent(text="middle"),
            second,
            types.TextContent(text="last"),
        ]
    )
    tool, connection = make_tool(stdio_config, result=source)
    registry = ToolRegistry()
    registry.register(tool)
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="mixed-call", name=tool.name, input={}),
        ToolExecutionContext(workspace_root=tmp_path, session_id="session/图片"),
    )
    assert not result.is_error and result.tool_use_id == "mixed-call"
    assert [part.type for part in result.content] == [
        "text",
        "image",
        "text",
        "image",
        "text",
        "text",
    ]
    images = [part for part in result.content if isinstance(part, ImageBlock)]
    assert [image.original.path.read_bytes() for image in images] == [
        base64.b64decode(first.data),
        base64.b64decode(second.data),
    ]
    assert images[0].original.path != images[1].original.path
    assert all(
        image.model.path.parent
        == tmp_path / ".iris" / "image-cache" / safe_path_segment("session/图片")
        for image in images
    )
    assert result.to_msg().tool_results[0].content == result.model_blocks
    completed = project_live_fact(
        RuntimeStreamEvent(
            kind="tool.completed",
            run_id="run",
            session_id="session/图片",
            activation_id="act",
            step_index=1,
            tool_call_id=result.tool_use_id,
            tool_name=tool.name,
            tool_ordinal=1,
            tool_result=result,
        )
    )[0]
    assert completed.payload["tool_call_id"] == "mixed-call"
    assert completed.payload["content"].count("[image]") == 2
    assert first.data not in json.dumps(completed.payload)
    assert len(connection.calls) == 1
    assert json.loads(result.artifact.path.read_text(encoding="utf-8")) == source.model_dump(
        mode="json", by_alias=True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("with_text", [False, True])
async def test_mcp_error_retains_images_with_one_authoritative_error(
    stdio_config: MCPResolvedServer, tmp_path: Path, with_text: bool
) -> None:
    first, second = _image_content("red"), _image_content("blue")
    content = [first, types.TextContent(text="failed"), second] if with_text else [first, second]
    source = types.CallToolResult(content=content, is_error=True)
    tool, connection = make_tool(stdio_config, result=source)
    registry = ToolRegistry()
    registry.register(tool)
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="error-images", name=tool.name, input={}),
        ToolExecutionContext(workspace_root=tmp_path, session_id="session"),
    )
    assert result.is_error and result.error.code == "MCP_TOOL_ERROR"
    assert result.model_content.count("Error[MCP_TOOL_ERROR]") == 1
    assert ("failed" if with_text else "MCP 工具返回业务错误") in result.error.message
    assert str(result.artifact.path) in result.error.message
    blocks = result.to_msg().tool_results[0].content
    assert [block.type for block in blocks] == ["text", "image", "image"]
    assert [
        block.original.path.read_bytes() for block in blocks if isinstance(block, ImageBlock)
    ] == [base64.b64decode(first.data), base64.b64decode(second.data)]
    assert len(connection.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["base64", "image", "save"])
async def test_local_image_failure_is_known_once_without_remote_replay(
    stdio_config: MCPResolvedServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    source_image = _image_content()
    if failure == "base64":
        source_image.data = "not-base64!"
    elif failure == "image":
        source_image.data = "YWJj"
    else:

        def fail_save(*args: Any, **kwargs: Any) -> ImageBlock:
            raise IrisImageError("图片保存失败")

        monkeypatch.setattr("iris.mcp.tools.import_tool_image", fail_save)
    tool, connection = make_tool(
        stdio_config, trust=False, result=types.CallToolResult(content=[source_image])
    )
    registry = ToolRegistry()
    registry.register(tool)
    result = await ToolExecutor(registry, permission_policy=AllowTools()).execute_one(
        ToolUseBlock(id="failed-image", name=tool.name, input={}),
        ToolExecutionContext(workspace_root=tmp_path, session_id="session"),
    )
    assert result.is_error and result.error.code == "IMAGE_ERROR"
    assert result.model_content.count("Error[IMAGE_ERROR]") == 1
    assert result.artifact is None
    assert not any(isinstance(block, ImageBlock) for block in result.content)
    assert len(connection.calls) == 1


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
