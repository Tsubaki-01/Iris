from __future__ import annotations

import pytest

from iris.exceptions import IrisToolValidationError
from iris.tools import ToolCapability, ToolRegistry, tool


def test_function_registration_exports_schema_from_type_hints() -> None:
    def greet(name: str, excited: bool = False) -> str:
        """生成问候语。"""
        suffix = "!" if excited else "."
        return f"你好，{name}{suffix}"

    registry = ToolRegistry()

    registry.register_function(greet, description="生成问候语")

    assert registry.active_schemas() == [
        {
            "name": "greet",
            "description": "生成问候语",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "excited": {"type": "boolean", "default": False},
                },
                "required": ["name"],
            },
        }
    ]


def test_deferred_tool_is_hidden_until_view_allows_it() -> None:
    def hidden(query: str) -> str:
        return query

    registry = ToolRegistry()
    registry.register_function(hidden, description="隐藏工具", deferred=True)

    assert registry.active_schemas() == []
    assert registry.view(allow={"hidden"}).active_schemas()[0]["name"] == "hidden"


def test_deny_takes_precedence_over_allow() -> None:
    def hidden(query: str) -> str:
        return query

    registry = ToolRegistry()
    registry.register_function(hidden, description="隐藏工具", deferred=True)

    view = registry.view(allow={"hidden"}, deny={"hidden"})

    assert view.active_schemas() == []


def test_duplicate_names_and_alias_conflicts_raise_validation_error() -> None:
    def first() -> str:
        return "first"

    def second() -> str:
        return "second"

    registry = ToolRegistry()
    registered = registry.register_function(first, description="第一个工具")
    registered.definition.aliases = ("alias",)

    with pytest.raises(IrisToolValidationError):
        registry.register_function(second, name="first", description="重复名称")

    with pytest.raises(IrisToolValidationError):
        registry.register_function(second, name="alias", description="别名冲突")


def test_registry_view_filters_by_group() -> None:
    def read_item(item_id: str) -> str:
        return item_id

    def write_item(item_id: str) -> str:
        return item_id

    registry = ToolRegistry()
    registry.register_function(
        read_item,
        description="读取条目",
        capabilities={ToolCapability.READ},
        group="readers",
    )
    registry.register_function(
        write_item,
        description="写入条目",
        capabilities={ToolCapability.WRITE},
        group="writers",
    )

    active_names = [
        schema["name"]
        for schema in registry.view(include_groups={"readers"}).active_schemas()
    ]

    assert active_names == ["read_item"]


def test_tool_decorator_metadata_is_used_when_registering_function() -> None:
    @tool(
        name="decorated_search",
        description="装饰器工具",
        capabilities={ToolCapability.NETWORK},
        group="network",
        deferred=True,
    )
    def search(query: str) -> str:
        return query

    registry = ToolRegistry()
    registered = registry.register_function(search)

    assert registered.definition.name == "decorated_search"
    assert registered.definition.description == "装饰器工具"
    assert registered.definition.capabilities == {ToolCapability.NETWORK}
    assert registered.definition.group == "network"
    assert registered.definition.deferred is True
    assert registry.active_schemas() == []
    assert registry.view(allow={"decorated_search"}).active_schemas()[0]["name"] == (
        "decorated_search"
    )
