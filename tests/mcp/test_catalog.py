"""目录的身份、schema 和本地信任规则。"""

from copy import deepcopy
from dataclasses import replace

import pytest
from mcp import types

from iris.mcp.catalog import build_catalog
from iris.mcp.models import MCPResolvedServer


def test_names_are_stable_and_original_schema_is_preserved(stdio_config: MCPResolvedServer) -> None:
    tools = [
        types.Tool(name=name, input_schema={"type": "object"})
        for name in ("echo", "a-b", "a.b", "x" * 80)
    ]
    original = deepcopy(tools[0].input_schema)
    catalog, diagnostics = build_catalog(stdio_config, tools)
    reverse, _ = build_catalog(stdio_config, list(reversed(tools)))
    assert not diagnostics
    names = {item.wire_name: item.public_name for item in catalog}
    assert names["echo"] == "mcp__fixture__echo"
    assert len(set(names.values())) == 4
    assert all(len(name) <= 64 for name in names.values())
    assert names == {item.wire_name: item.public_name for item in reverse}
    assert tools[0].input_schema == original
    assert catalog[0].definition.description == "MCP tool echo from fixture"


def test_unicode_identity_gets_provider_compatible_public_name(
    stdio_config: MCPResolvedServer,
) -> None:
    """中文配置身份不能使整个 provider 工具目录不可用。"""
    catalog, diagnostics = build_catalog(
        replace(stdio_config, server_id="中文服务"),
        [
            types.Tool(name="查询", input_schema={"type": "object"}),
        ],
    )
    assert not diagnostics
    descriptor = catalog[0]
    assert descriptor.public_name.isascii() and len(descriptor.public_name) <= 64
    assert (descriptor.server_id, descriptor.wire_name) == ("中文服务", "查询")


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "$schema": "http://json-schema.org/draft-07/schema#"},
        {"type": "object", "properties": {"x": {"type": "invalid"}}},
        {"type": "object", "properties": {"x": {"$ref": "https://example.test/schema"}}},
        {"type": "object", "$ref": "#/$defs/missing"},
    ],
)
def test_unsupported_schemas_are_excluded(stdio_config: MCPResolvedServer, schema: dict) -> None:
    catalog, diagnostics = build_catalog(
        stdio_config, [types.Tool(name="bad", input_schema=schema)]
    )
    assert catalog == () and diagnostics[0].wire_name == "bad"


def test_composition_and_local_refs(stdio_config: MCPResolvedServer) -> None:
    schema = {
        "type": "object",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {"value": {"type": "integer"}},
        "properties": {"x": {"$ref": "#/$defs/value"}},
        "allOf": [{"required": ["x"]}],
        "examples": [{"$ref": "literal instance data"}],
    }
    catalog, diagnostics = build_catalog(
        stdio_config, [types.Tool(name="valid", input_schema=schema)]
    )
    assert not diagnostics and len(catalog) == 1
    assert catalog[0].input_validator.is_valid({"x": 1})
    assert not catalog[0].input_validator.is_valid({"x": "1"})
    assert catalog[0].definition.input_schema["allOf"] == [{"required": ["x"]}]


def test_filters_apply_before_schema_validation(stdio_config: MCPResolvedServer) -> None:
    tools = [
        types.Tool(name="bad", input_schema={"type": "invalid"}),
        types.Tool(name="good", input_schema={"type": "object"}),
    ]
    for config in (
        replace(stdio_config, enabled_tools=("good",)),
        replace(stdio_config, disabled_tools=("bad",)),
    ):
        catalog, diagnostics = build_catalog(config, tools)
        assert [item.wire_name for item in catalog] == ["good"] and not diagnostics
    catalog, _ = build_catalog(replace(stdio_config, enabled_tools=()), tools)
    assert catalog == ()


@pytest.mark.parametrize(
    ("trust", "hint", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (True, None, False),
        (False, True, False),
        (False, False, False),
        (False, None, False),
    ],
)
def test_annotation_requires_explicit_local_trust(
    stdio_config: MCPResolvedServer, trust: bool, hint: bool | None, expected: bool
) -> None:
    tool = types.Tool(
        name="read",
        input_schema={"type": "object"},
        annotations=types.ToolAnnotations(read_only_hint=hint) if hint is not None else None,
    )
    catalog, _ = build_catalog(replace(stdio_config, trust_annotations=trust), [tool])
    assert catalog[0].trusted_read_only is expected
