"""工具最终化固化声明，不让历史保留策略依赖后续 registry。"""

from pathlib import Path

import pytest

from iris.message import TextBlock, ToolUseBlock
from iris.tools import CallableTool, ToolExecutionContext, ToolExecutor, ToolRegistry, ToolResult
from iris.tools.builtin.file import GrepSearchTool, ListFilesTool, ReadFileTool, WriteFileTool
from iris.tools.builtin.web import WebFetchTool, WebSearchTool


@pytest.mark.asyncio
async def test_finalization_owns_retention_and_canonical_name(tmp_path: Path) -> None:
    """别名保留调用身份，canonical name 与声明覆盖用户同名 metadata。"""
    source = ToolResult(
        tool_use_id="",
        tool_name="alias",
        content=[TextBlock(text="body")],
        metadata={
            "context_retention": "observation",
            "context_tool_name": "wrong",
            "extra": {
                "context_retention": "observation",
                "context_tool_name": "also_wrong",
                "note": 1,
            },
        },
    )

    def observe() -> ToolResult:
        """返回普通工具内容。"""
        return source

    tool = CallableTool(observe, name="canonical")
    tool.definition.aliases = ("alias",)
    registry = ToolRegistry()
    registry.register(tool)
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="call", name="alias"), ToolExecutionContext(workspace_root=tmp_path)
    )
    block = result.to_msg().tool_results[0]
    assert block.tool_use_id == "call" and block.name == "alias"
    assert block.metadata["extra"] == {
        "context_retention": "keep",
        "context_tool_name": "canonical",
        "note": 1,
    }
    tool.definition.context_retention = "observation"
    assert block.metadata["extra"]["context_retention"] == "keep"
    assert source.metadata["context_retention"] == "observation"


def test_builtin_observations_are_explicit_and_writes_stay_keep() -> None:
    """只给已明确的文件/网页观察工具声明 observation。"""
    for tool in [
        ReadFileTool(),
        ListFilesTool(),
        GrepSearchTool(),
        WebSearchTool(api_key="test"),
        WebFetchTool(api_key="test"),
    ]:
        assert tool.definition.context_retention == "observation"
    assert WriteFileTool().definition.context_retention == "keep"
