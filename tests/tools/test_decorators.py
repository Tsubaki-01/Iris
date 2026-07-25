from __future__ import annotations

from iris.tools import ToolRegistry, tool


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
