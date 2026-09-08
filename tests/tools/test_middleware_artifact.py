"""工具最终结果在全部 middleware 后统一生成 artifact。"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from iris.message import TextBlock, ToolUseBlock
from iris.tools import (
    BaseTool,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolRegistry,
    ToolResult,
)


@pytest.mark.asyncio
async def test_after_call_expansion_is_persisted_once(tmp_path: Path) -> None:
    """正常扩展结果的 hook 也遵守工具声明的长度阈值。"""

    class ExpandResult(ToolMiddleware):
        """为结果补充描述。"""

        async def after_call(
            self, tool: BaseTool, result: ToolResult, context: ToolExecutionContext
        ) -> ToolResult:
            """补充正文，保持工具 identity。"""
            return result.model_copy(update={"content": [TextBlock(text="expanded result")]})

    class SmallTool(BaseTool):
        """声明短结果阈值的正常工具。"""

        definition = ToolDefinition(
            name="small",
            description="短结果",
            input_schema={"type": "object", "properties": {}},
            max_result_chars=3,
        )

        async def arun(
            self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
        ) -> ToolResult:
            """返回阈值内的原始结果。"""
            return ToolResult(
                tool_use_id=context.call_id, tool_name=self.name, content=[TextBlock(text="ok")]
            )

    registry = ToolRegistry()
    registry.register(SmallTool())
    result = await ToolExecutor(
        registry, middleware=[ExpandResult()], artifact_preview_chars=2
    ).execute_one(
        ToolUseBlock(id="call", name="small", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert not result.is_error
    assert result.artifact is not None
    assert result.artifact.path.read_text(encoding="utf-8") == "expanded result"
    assert result.artifact.preview == "ex"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_id", "second_id"),
    [("call/1", "call_1"), ("Call", "call"), ("中文", "__"), ("", "default")],
)
@pytest.mark.parametrize("id_field", ["call", "session"])
async def test_different_ids_keep_both_artifacts(
    tmp_path: Path, first_id: str, second_id: str, id_field: str
) -> None:
    """不同调用或会话 ID 的大结果同时可读，路径仍位于 artifact 根目录。"""

    def echo(value: str) -> str:
        return value

    registry = ToolRegistry()
    tool = registry.register_function(echo)
    tool.definition.max_result_chars = 3
    executor = ToolExecutor(registry)
    artifacts = []
    for identifier, content in [(first_id, "first result"), (second_id, "second result")]:
        result = await executor.execute_one(
            ToolUseBlock(
                id=identifier if id_field == "call" else "same-call",
                name="echo",
                input={"value": content},
            ),
            ToolExecutionContext(
                workspace_root=tmp_path,
                session_id=identifier if id_field == "session" else "same-session",
            ),
        )
        assert not result.is_error
        assert result.artifact is not None
        result.artifact.path.relative_to(tmp_path / ".iris" / "tool-results")
        artifacts.append(result.artifact)

    assert artifacts[0].path != artifacts[1].path
    assert artifacts[0].path.read_text(encoding="utf-8") == "first result"
    assert artifacts[1].path.read_text(encoding="utf-8") == "second result"
