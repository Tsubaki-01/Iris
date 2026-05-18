from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris.message import ToolUseBlock
from iris.tools import (
    CircuitBreaker,
    DocstringSchemaExtractor,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    ToolSearchTool,
    tool,
)


def test_enhanced_tool_decorator_metadata_is_preserved() -> None:
    @tool(
        name="legacy_lookup",
        description="查询旧索引",
        deferred=True,
        preset_kwargs={"token": "secret"},
        examples=[{"input": {"query": "iris"}, "output": "ok"}],
        tags=["search", "legacy"],
        version="1.2.0",
        deprecated=True,
        deprecation_message="请改用 lookup_v2",
    )
    def lookup(query: str, token: str) -> str:
        return f"{query}:{token}"

    registry = ToolRegistry()
    registered = registry.register_function(lookup)

    assert registered.definition.input_schema["properties"] == {"query": {"type": "string"}}
    assert registered.definition.metadata == {
        "examples": [{"input": {"query": "iris"}, "output": "ok"}],
        "tags": ["search", "legacy"],
        "version": "1.2.0",
        "deprecated": True,
        "deprecation_message": "请改用 lookup_v2",
    }


def test_docstring_args_are_added_to_callable_schema() -> None:
    def search(query: str, limit: int = 5) -> str:
        """搜索资料。

        Args:
            query: 查询关键词。
            limit: 最多返回数量。
        """
        return query * limit

    registry = ToolRegistry()
    registered = registry.register_function(search)

    properties = registered.definition.input_schema["properties"]
    assert properties["query"]["description"] == "查询关键词。"
    assert properties["limit"]["description"] == "最多返回数量。"


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


@pytest.mark.asyncio
async def test_executor_runs_middleware_before_and_after_call(tmp_path: Path) -> None:
    class AuditMiddleware:
        async def before_call(
            self,
            tool: Any,
            params: dict[str, Any],
            context: ToolExecutionContext,
        ) -> None:
            context.metadata["before"] = f"{tool.name}:{params['name']}"

        async def after_call(
            self,
            tool: Any,
            result: ToolResult,
            context: ToolExecutionContext,
        ) -> ToolResult:
            return result.model_copy(
                update={
                    "metadata": {
                        **result.metadata,
                        "extra": {"audit": context.metadata["before"]},
                    }
                }
            )

    def greet(name: str) -> str:
        return f"你好，{name}"

    registry = ToolRegistry()
    registry.register_function(greet, description="生成问候语")

    result = await ToolExecutor(registry, middleware=[AuditMiddleware()]).execute_one(
        ToolUseBlock(id="call_1", name="greet", input={"name": "Iris"}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.metadata["extra"] == {"audit": "greet:Iris"}


@pytest.mark.asyncio
async def test_executor_maps_on_error_middleware_failure(tmp_path: Path) -> None:
    class BrokenErrorMiddleware:
        async def on_error(
            self,
            tool: Any,
            error: Exception,
            context: ToolExecutionContext,
        ) -> ToolResult | None:
            raise RuntimeError("middleware failed")

    def fail() -> str:
        raise RuntimeError("tool failed")

    registry = ToolRegistry()
    registry.register_function(fail, description="失败工具")

    result = await ToolExecutor(registry, middleware=[BrokenErrorMiddleware()]).execute_one(
        ToolUseBlock(id="call_1", name="fail", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.error is not None
    assert result.error.code == "MIDDLEWARE_ERROR"
    assert result.error.details["middleware_error"] == "middleware failed"
    assert "tool failed" in str(result.error.details["original_error"])


@pytest.mark.asyncio
async def test_executor_accepts_legacy_after_execute_only_middleware(tmp_path: Path) -> None:
    class LegacyMiddleware:
        async def after_execute(
            self,
            result: ToolResult,
            context: ToolExecutionContext,
        ) -> ToolResult:
            return result.model_copy(update={"metadata": {"extra": {"legacy": True}}})

    def greet() -> str:
        return "你好"

    registry = ToolRegistry()
    registry.register_function(greet, description="生成问候语")

    result = await ToolExecutor(registry, middleware=[LegacyMiddleware()]).execute_one(
        ToolUseBlock(id="call_1", name="greet", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.metadata["extra"] == {"legacy": True}


def test_docstring_schema_extractor_is_public() -> None:
    assert isinstance(DocstringSchemaExtractor(), DocstringSchemaExtractor)


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_consecutive_failures(tmp_path: Path) -> None:
    calls = 0

    def unstable() -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register_function(unstable, description="不稳定工具")
    executor = ToolExecutor(
        registry,
        circuit_breaker=CircuitBreaker(failure_threshold=2, cooldown_seconds=60),
    )

    for _ in range(2):
        result = await executor.execute_one(
            ToolUseBlock(id="call_1", name="unstable", input={}),
            ToolExecutionContext(workspace_root=tmp_path),
        )
        assert result.error is not None
        assert result.error.code == "EXECUTION_ERROR"

    result = await executor.execute_one(
        ToolUseBlock(id="call_2", name="unstable", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert calls == 2
    assert result.error is not None
    assert result.error.code == "CIRCUIT_OPEN"
