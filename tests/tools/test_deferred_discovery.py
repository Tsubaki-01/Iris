"""系统搜索从静态可发现目录排名，并固化成功发现事实。"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from iris.agents import ContextPolicyConfig
from iris.message import TextBlock, ToolUseBlock
from iris.tools import (
    CallableTool,
    ToolCall,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolNext,
    ToolRegistry,
    ToolResult,
)
from iris.tools.discovery import ToolSearchInput, ToolSearchTool


def test_deferred_policy_and_search_input() -> None:
    assert ContextPolicyConfig().deferred_tools is False
    with pytest.raises(ValueError, match="enabled"):
        ContextPolicyConfig(enabled=False, deferred_tools=True)
    assert ToolSearchInput(queries=[" docs "]).queries == ["docs"]
    with pytest.raises(ValueError):
        ToolSearchInput(queries=["docs"], limit=1)


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
        ToolUseBlock(
            id="find",
            name="tool_search",
            input={
                "queries": ["docs_allowed", "docs_exception", "docs_allowed", "unmatchedneedle"]
            },
        ),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    names = [item["name"] for item in result.data["tools"]]
    assert names == ["docs_allowed", "docs_exception"]
    assert result.data["selections"] == [
        {"query": "docs_allowed", "tool": "docs_allowed"},
        {"query": "docs_exception", "tool": "docs_exception"},
        {"query": "docs_allowed", "tool": "docs_allowed"},
        {"query": "unmatchedneedle", "tool": None},
    ]
    assert result.to_block_metadata()["extra"]["context_revealed_tools"] == names
    assert all(len(item["description"]) <= 240 for item in result.data["tools"])
    assert view.allow == {"docs_exception"}
    first = await registry.get("tool_search").arun(
        ToolSearchInput(queries=["docs_denied docs"]),
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

        async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
            try:
                await call_next()
            except RuntimeError:
                return ToolResult(
                    tool_use_id="",
                    tool_name="",
                    content=[TextBlock(text="substitute")],
                    metadata={"context_revealed_tools": ["not-executed"]},
                )
            return ToolResult(
                tool_use_id="",
                tool_name="",
                content=[TextBlock(text="formatted")],
                is_error=outcome == "final_error",
            )

    if outcome == "handled_error":

        async def fail(
            params: BaseModel | dict[str, Any], context: ToolExecutionContext
        ) -> ToolResult:
            raise RuntimeError("discovery unavailable")

        monkeypatch.setattr(search, "arun", fail)
    result = await ToolExecutor(registry, middleware=[Rewrite()]).execute_one(
        ToolUseBlock(id="search", name="tool_search", input={"queries": ["docs"]}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    facts = result.to_block_metadata()["extra"]
    if outcome == "rewritten":
        assert facts["context_revealed_tools"] == ["docs"]
        assert result.model_content == "formatted"
    else:
        assert "context_revealed_tools" not in facts


@pytest.mark.asyncio
@pytest.mark.parametrize("groups", [None, []])
async def test_local_nonword_query_is_a_valid_empty_selection(
    tmp_path: Path, groups: list[str] | None
) -> None:
    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="docs", description="docs", deferred=True)
    registry.register(ToolSearchTool(registry.view()))
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(
            id="find", name="tool_search", input={"queries": [" !!! "], "include_groups": groups}
        ),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert not result.is_error
    assert result.data == {"selections": [{"query": "!!!", "tool": None}], "tools": []}
    assert "decision" not in result.metadata


@pytest.mark.asyncio
async def test_local_search_calls_existing_index_once_per_query_with_limit_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="docs", description="docs", deferred=True)
    search = ToolSearchTool(registry.view())
    calls: list[tuple[str, dict[str, Any]]] = []
    original = registry.search_deferred

    def record(query: str, **kwargs: Any) -> list[Any]:
        calls.append((query, kwargs))
        return original(query, **kwargs)

    monkeypatch.setattr(registry, "search_deferred", record)
    await search.arun(
        ToolSearchInput(queries=["docs", "absent", "docs"], include_groups=["core"]),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert calls == [
        (query, {"include_groups": {"core"}, "limit": 1, "allowed_names": {"docs"}})
        for query in ["docs", "absent", "docs"]
    ]
