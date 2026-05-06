from __future__ import annotations

from pydantic import BaseModel

from iris.tools import (
    ToolDefinition,
    schema_from_pydantic_model,
    to_anthropic_tool_schema,
    to_openai_chat_tool_schema,
    to_openai_responses_tool_schema,
)


def test_provider_schema_shapes_are_locked() -> None:
    definition = ToolDefinition(
        name="search",
        description="搜索资料",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    assert to_openai_chat_tool_schema(definition) == {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索资料",
            "parameters": definition.input_schema,
        },
    }
    assert to_openai_responses_tool_schema(definition) == {
        "type": "function",
        "name": "search",
        "description": "搜索资料",
        "parameters": definition.input_schema,
        "strict": False,
    }
    assert to_anthropic_tool_schema(definition) == {
        "name": "search",
        "description": "搜索资料",
        "input_schema": definition.input_schema,
    }


def test_schema_from_pydantic_model_preserves_nested_model_defs() -> None:
    class SearchOptions(BaseModel):
        limit: int

    class SearchInput(BaseModel):
        query: str
        options: SearchOptions

    schema = schema_from_pydantic_model(SearchInput)

    assert "$defs" in schema
    assert schema["properties"]["options"] == {"$ref": "#/$defs/SearchOptions"}
