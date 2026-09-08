"""Runtime environment 的确定性恢复指纹。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic_core import PydanticSerializationError

from ..exceptions import IrisConfigError, IrisContextError
from ..lifecycle import validate_json_safe
from ..runtime import AgentRuntime
from ..tools import ToolDefinition


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
