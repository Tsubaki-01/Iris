"""系统搜索从静态可发现目录排名，并固化成功发现事实。"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from iris.agents import ContextPolicyConfig
from iris.message import TextBlock, ToolUseBlock
from iris.tools import (
    BaseTool,
    CallableTool,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolRegistry,
    ToolResult,
)
from iris.tools.discovery import ToolSearchInput, ToolSearchTool


def test_deferred_policy_and_search_limits() -> None:
    assert ContextPolicyConfig().deferred_tools is False
    with pytest.raises(ValueError, match="enabled"):
        ContextPolicyConfig(enabled=False, deferred_tools=True)
    assert ToolSearchInput(query="docs").limit == 3
    with pytest.raises(ValueError):
        ToolSearchInput(query="docs", limit=21)


@pytest.mark.asyncio
async def test_search_filters_before_ranking_and_saves_canonical_names(tmp_path: Path) -> None:
    registry = ToolRegistry()
    for name, group in (
        ("docs_denied", "allowed"),
        ("docs_outside", "outside"),
        ("docs_exception", "outside"),
        ("docs_allowed", "allowed"),
    ):
        registry.register_function(
            lambda: "ok", name=name, description="docs " * 100, group=group, deferred=True
        )
    view = registry.view(deny={"docs_denied"}, include_groups={"allowed"}, allow={"docs_exception"})
    registry.register(ToolSearchTool(view))
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="find", name="tool_search", input={"query": "docs", "limit": 20}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    names = [item["name"] for item in result.data["tools"]]
    assert set(names) == {"docs_allowed", "docs_exception"}
    assert result.to_block_metadata()["extra"]["context_revealed_tools"] == names
    assert all(len(item["description"]) <= 240 for item in result.data["tools"])
    assert view.allow == {"docs_exception"}
    first = await registry.get("tool_search").arun(
        ToolSearchInput(query="docs_denied docs", limit=1),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert len(first.data["tools"]) == 1 and first.data["tools"][0]["name"] in names


@pytest.mark.asyncio
async def test_other_tool_metadata_does_not_create_disclosure(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(
        CallableTool(
            lambda: ToolResult(
                tool_use_id="",
                tool_name="other",
                content=[TextBlock(text="ok")],
                metadata={
                    "context_revealed_tools": ["secret"],
                    "extra": {"context_revealed_tools": ["secret"]},
                },
            ),
            name="other",
            description="other",
        )
    )
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="other", name="other"), ToolExecutionContext(workspace_root=tmp_path)
    )
    assert "context_revealed_tools" not in result.to_block_metadata()["extra"]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["rewritten", "final_error", "handled_error"])
async def test_search_disclosure_uses_successful_body_not_middleware_metadata(
    tmp_path: Path, outcome: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="docs", description="docs", deferred=True)
    search = ToolSearchTool(registry.view())
    registry.register(search)

    class Rewrite(ToolMiddleware):
        """模拟正常正文重建和错误替代。"""

        async def after_call(
            self, tool: BaseTool, result: ToolResult, context: ToolExecutionContext
        ) -> ToolResult:
            return ToolResult(
                tool_use_id="",
                tool_name="",
                content=[TextBlock(text="formatted")],
                is_error=outcome == "final_error",
            )

        async def on_error(
            self, tool: BaseTool, error: Exception, context: ToolExecutionContext
        ) -> ToolResult:
            return ToolResult(
                tool_use_id="",
                tool_name="",
                content=[TextBlock(text="substitute")],
                metadata={"context_revealed_tools": ["not-executed"]},
            )

    if outcome == "handled_error":

        async def fail(
            params: BaseModel | dict[str, Any], context: ToolExecutionContext
        ) -> ToolResult:
            raise RuntimeError("discovery unavailable")

        monkeypatch.setattr(search, "arun", fail)
    result = await ToolExecutor(registry, middleware=[Rewrite()]).execute_one(
        ToolUseBlock(id="search", name="tool_search", input={"query": "docs"}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    facts = result.to_block_metadata()["extra"]
    if outcome == "rewritten":
        assert facts["context_revealed_tools"] == ["docs"]
        assert result.model_content == "formatted"
    else:
        assert "context_revealed_tools" not in facts
