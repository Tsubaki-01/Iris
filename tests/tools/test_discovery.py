from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from iris.exceptions import IrisToolValidationError
from iris.message import ToolUseBlock
from iris.tools import (
    BaseTool,
    DeferredToolIndex,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolSearchTool,
    tool,
)


@pytest.mark.asyncio
async def test_tool_search_returns_deferred_matches_without_activating_them(
    tmp_path: Path,
) -> None:
    @tool(deferred=True, tags=["embedding"], group="research")
    def vector_lookup(query: str) -> str:
        """检索向量知识库。"""
        return query

    registry = ToolRegistry()
    registry.register_function(vector_lookup)
    registry.register(ToolSearchTool(registry))

    assert [schema["name"] for schema in registry.active_schemas()] == ["tool_search"]

    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="call_1", name="tool_search", input={"query": "embedding"}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is False
    assert result.data["tools"][0]["name"] == "vector_lookup"
    assert [schema["name"] for schema in registry.active_schemas()] == ["tool_search"]


def test_deferred_tool_search_uses_bm25_like_ranking() -> None:
    @tool(
        name="alpha_dense",
        description="search search search search documents",
        deferred=True,
        tags=["text"],
        group="research",
    )
    def alpha_dense(query: str) -> str:
        return query

    @tool(
        name="vector_lookup",
        description="Find relevant embeddings for semantic retrieval",
        deferred=True,
        tags=["vector", "embedding"],
        group="research",
    )
    def vector_lookup(query: str) -> str:
        return query

    @tool(
        name="file_search",
        description="Search files",
        deferred=True,
        tags=["file"],
        group="file",
    )
    def file_search(query: str) -> str:
        return query

    registry = ToolRegistry()
    registry.register_function(alpha_dense)
    registry.register_function(vector_lookup)
    registry.register_function(file_search)
    index = DeferredToolIndex()
    index.build(registry.view(allow={"alpha_dense", "vector_lookup", "file_search"}).active_tools)

    assert index.naive_search("dense")[0].name == "alpha_dense"
    assert index.search("vector search")[0].name == "vector_lookup"


def test_deferred_tool_search_handles_empty_index_and_invalid_query() -> None:
    index = DeferredToolIndex()

    assert index.search("anything") == []
    with pytest.raises(IrisToolValidationError):
        index.search("   ")


def test_deferred_tool_search_supports_groups_limit_and_stable_sorting() -> None:
    registry = ToolRegistry()

    for index in range(60):
        @tool(
            name=f"bulk_tool_{index:02d}",
            description="shared marker",
            deferred=True,
            group="bulk",
        )
        def bulk_tool(query: str) -> str:
            return query

        registry.register_function(bulk_tool)

    @tool(name="alpha_tool", description="stable marker", deferred=True, group="alpha")
    def alpha_tool(query: str) -> str:
        return query

    @tool(name="beta_tool", description="stable marker", deferred=True, group="beta")
    def beta_tool(query: str) -> str:
        return query

    registry.register_function(alpha_tool)
    registry.register_function(beta_tool)

    bulk_results = registry.search_deferred("shared", limit=500)
    assert len(bulk_results) == 50

    alpha_results = registry.search_deferred("stable", include_groups={"alpha"})
    assert [definition.name for definition in alpha_results] == ["alpha_tool"]

    stable_results = registry.search_deferred("stable")
    assert [definition.name for definition in stable_results] == ["alpha_tool", "beta_tool"]


def test_registry_search_deferred_reuses_cached_index_until_registry_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls = 0
    original_build = DeferredToolIndex.build

    def counted_build(self: DeferredToolIndex, tools: Iterable[BaseTool]) -> None:
        nonlocal build_calls
        build_calls += 1
        original_build(self, tools)

    monkeypatch.setattr(DeferredToolIndex, "build", counted_build)

    @tool(name="alpha_tool", description="stable marker", deferred=True, group="alpha")
    def alpha_tool(query: str) -> str:
        return query

    @tool(name="beta_tool", description="stable marker", deferred=True, group="beta")
    def beta_tool(query: str) -> str:
        return query

    registry = ToolRegistry()
    registry.register_function(alpha_tool)

    first = registry.search_deferred("stable")
    second = registry.search_deferred("stable")

    assert [definition.name for definition in first] == ["alpha_tool"]
    assert [definition.name for definition in second] == ["alpha_tool"]
    assert build_calls == 1

    registry.register_function(beta_tool)
    third = registry.search_deferred("stable")

    assert [definition.name for definition in third] == ["alpha_tool", "beta_tool"]
    assert build_calls == 2


def test_deferred_tool_search_matches_chinese_description_terms() -> None:
    @tool(
        name="vector_lookup",
        description="检索向量知识库",
        deferred=True,
        tags=["embedding"],
        group="research",
    )
    def vector_lookup(query: str) -> str:
        return query

    registry = ToolRegistry()
    registry.register_function(vector_lookup)

    matches = registry.search_deferred("向量 检索")

    assert [definition.name for definition in matches] == ["vector_lookup"]


def test_deferred_tool_search_prefers_cjk_bigram_and_query_coverage() -> None:
    @tool(
        name="vector_lookup",
        description="检索向量知识库",
        deferred=True,
        group="research",
    )
    def vector_lookup(query: str) -> str:
        return query

    @tool(
        name="vector_index",
        description="向量索引",
        deferred=True,
        tags=["向量"],
        group="research",
    )
    def vector_index(query: str) -> str:
        return query

    registry = ToolRegistry()
    registry.register_function(vector_lookup)
    registry.register_function(vector_index)

    matches = registry.search_deferred("向量 检索")

    assert [definition.name for definition in matches] == ["vector_lookup", "vector_index"]


def test_deferred_tool_search_uses_low_weight_cjk_single_character_recall() -> None:
    @tool(
        name="knowledge_base",
        description="知识库检索",
        deferred=True,
        group="research",
    )
    def knowledge_base(query: str) -> str:
        return query

    @tool(
        name="knowledge_graph",
        description="知识图谱",
        deferred=True,
        group="research",
    )
    def knowledge_graph(query: str) -> str:
        return query

    registry = ToolRegistry()
    registry.register_function(knowledge_base)
    registry.register_function(knowledge_graph)

    assert registry.search_deferred("库")[0].name == "knowledge_base"
    assert registry.search_deferred("知识库")[0].name == "knowledge_base"
