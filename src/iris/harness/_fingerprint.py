"""Runtime environment 的确定性恢复指纹。"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

from pydantic_core import PydanticSerializationError

from ..exceptions import IrisConfigError, IrisContextError
from ..lifecycle import validate_json_safe
from ..runtime import AgentRuntime
from ..tools import ToolDefinition

if TYPE_CHECKING:
    from ..mcp.models import MCPCatalogSnapshot


def _mcp_payload(snapshot: MCPCatalogSnapshot) -> list[dict[str, Any]]:
    """补充 ToolDefinition 未涵盖的 MCP 有效契约；调用方已经完成 prepare。"""
    servers: list[dict[str, Any]] = []
    for server in sorted(snapshot.servers, key=lambda item: item.config.server_id):
        config = server.config
        servers.append(
            {
                "server_id": config.server_id,
                "transport": config.transport,
                "protocol_version": server.protocol_version,
                "command": config.command,
                "args": list(config.args),
                "cwd": str(config.cwd) if config.cwd is not None else None,
                "env": config.env,
                "url": config.url,
                "headers": config.headers,
                "required": config.required,
                "trust_annotations": config.trust_annotations,
                "startup_timeout_sec": config.startup_timeout_sec,
                "tool_timeout_sec": config.tool_timeout_sec,
                "enabled_tools": sorted(set(config.enabled_tools))
                if config.enabled_tools is not None
                else None,
                "disabled_tools": sorted(set(config.disabled_tools)),
                "tools": [
                    {
                        "wire_name": tool.wire_name,
                        "output_schema": tool.sdk_tool.output_schema,
                        "trusted_read_only": tool.trusted_read_only,
                    }
                    for tool in sorted(server.tools, key=lambda item: item.wire_name)
                ],
            }
        )
    return servers


def _tool_payload(definition: ToolDefinition) -> dict[str, Any]:
    """把工具定义归一化为不依赖 set 迭代顺序的 JSON 对象。"""
    payload = definition.model_dump(mode="json")
    payload["aliases"] = list(definition.aliases)
    payload["capabilities"] = sorted(capability.value for capability in definition.capabilities)
    return payload


def compute_environment_fingerprint(runtime: AgentRuntime) -> str:
    """计算 checkpoint 恢复兼容性所需的 canonical SHA-256。"""
    environment = runtime.environment
    model = environment.agent_config.model
    skills = environment.skill_registry
    try:
        policy_payload = (
            environment.tool_bridge.tool_executor.permission_policy.fingerprint_payload()
        )
        payload = {
            "agent_name": environment.agent_config.name,
            "model": {
                "name": model.name,
                "request_options": model.to_llm_request_options(),
            },
            "provider": environment.provider_fingerprint,
            "context": environment.context_builder.fingerprint_payload(environment.context_input),
            "skills": (
                {name: skills.get(name).content_version for name in skills.names()}
                if skills is not None
                else {}
            ),
            "tools": [
                _tool_payload(tool.definition)
                for tool in sorted(
                    environment.tool_bridge.tool_view.active_tools,
                    key=lambda item: item.definition.name,
                )
            ],
            "permission_policy": policy_payload,
            "workspace_root": str(environment.workspace_root.resolve()),
            "checkpoint_version": 1,
        }
        if environment.mcp_manager is not None:
            payload["mcp"] = _mcp_payload(
                cast("MCPCatalogSnapshot", environment.mcp_manager.snapshot)
            )
        validate_json_safe(payload, field_name="environment fingerprint payload")
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (IrisConfigError, IrisContextError):
        raise
    except (PydanticSerializationError, TypeError, ValueError) as exc:
        raise IrisConfigError("environment fingerprint payload 不是 JSON-safe") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["compute_environment_fingerprint"]
