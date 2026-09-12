"""将完整 SDK 目录构造成 Iris 工具描述，不连接或注册工具。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator, SchemaError
from mcp import types
from pydantic import ValidationError
from referencing import Registry
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

from ..exceptions import IrisToolValidationError
from ..tools.base import ToolCapability, ToolDefinition
from .models import MCPDiagnostic, MCPResolvedServer, MCPToolDescriptor

_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def _public_name(server_id: str, wire_name: str) -> str:
    """只在名称需要转换或截断时用原始身份摘要消除歧义。"""
    base = f"mcp__{server_id}__{wire_name}"
    normalized = "".join(char if char.isalnum() or char == "_" else "_" for char in base)
    if normalized == base and len(base) <= 64:
        return base
    identity = json.dumps([server_id, wire_name], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    return f"{normalized[:52]}__{digest}"


def _compile_input_schema(schema: dict[str, Any]) -> Draft202012Validator:
    """按 2020-12 编译，并在目录边界确认引用均可本地解析。"""
    Draft202012Validator.check_schema(schema)
    resource = DRAFT202012.create_resource(schema)
    registry = Registry().with_resource("", resource)
    pending = [(resource, registry.resolver_with_root(resource))]
    while pending:
        current, resolver = pending.pop()
        contents = current.contents
        if isinstance(contents, Mapping):
            if contents.get("$schema", _DIALECT).rstrip("#") != _DIALECT:
                raise IrisToolValidationError("仅支持 JSON Schema 2020-12 dialect")
            for keyword in ("$ref", "$dynamicRef"):
                if keyword in contents:
                    resolver.lookup(contents[keyword])
        # 使用 dialect 的子 schema 遍历，避免将 examples/const 内的实例数据当 schema。
        pending.extend((child, resolver.in_subresource(child)) for child in current.subresources())
    return Draft202012Validator(schema, registry=registry)


def build_catalog(
    server: MCPResolvedServer,
    tools: Sequence[types.Tool],
) -> tuple[tuple[MCPToolDescriptor, ...], tuple[MCPDiagnostic, ...]]:
    """过滤原始工具名，构造 descriptor；无效单工具留下诊断后排除。

    Args:
        server: 已求值的 server 配置。
        tools: 同一 session 完整分页发现的 SDK 工具。

    Returns:
        可发布的 descriptors 与单工具诊断；注册冲突由 registry 负责。
    """
    descriptors: list[MCPToolDescriptor] = []
    diagnostics: list[MCPDiagnostic] = []
    for tool in tools:
        if tool.name in server.disabled_tools or (
            server.enabled_tools is not None and tool.name not in server.enabled_tools
        ):
            continue
        try:
            validator = _compile_input_schema(tool.input_schema)
            name = _public_name(server.server_id, tool.name)
            definition = ToolDefinition(
                name=name,
                description=tool.description or f"MCP tool {tool.name} from {server.server_id}",
                input_schema=deepcopy(tool.input_schema),
                capabilities={ToolCapability.MCP},
                group="mcp",
            )
        except (SchemaError, Unresolvable, IrisToolValidationError, ValidationError) as error:
            message = "工具 schema 或描述无效，或包含无法本地解析的引用"
            if isinstance(error, IrisToolValidationError):
                message = error.message
            diagnostics.append(
                MCPDiagnostic(
                    server.server_id,
                    "catalog",
                    "MCP_TOOL_INVALID",
                    message,
                    "input_schema",
                    tool.name,
                )
            )
            continue
        descriptors.append(
            MCPToolDescriptor(
                server.server_id,
                tool.name,
                name,
                tool,
                definition,
                validator,
                server.trust_annotations
                and tool.annotations is not None
                and tool.annotations.read_only_hint is True,
            )
        )
    return tuple(descriptors), tuple(diagnostics)


__all__ = ["build_catalog"]
