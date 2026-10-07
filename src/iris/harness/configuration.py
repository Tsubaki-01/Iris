"""Runner 已采用的配置与依赖描述，不从工作区重建过去来源。"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from ..agents import AgentConfig
from ..skill.models import SkillMetadata
from ..tools import ToolDefinition
from ..tools.registry import ToolRegistryView
from ..utils.sources import SourceDocument


@dataclass(frozen=True, slots=True)
class ConfigurationDependency:
    """实际依赖类型及装配来源；不序列化凭据或 Python 对象内部状态。"""

    kind: str
    type_name: str
    origin: Literal["configured", "injected"]


@dataclass(frozen=True, slots=True)
class LifecycleStorageDescription:
    """当前实际 store 身份，包含显式注入覆盖后的 backend/path。"""

    backend: str
    source_id: str
    path: str | None


@dataclass(frozen=True, slots=True)
class EffectiveConfiguration:
    """构造期配置、冻结来源和准备后实际目录的只读描述。"""

    configuration_snapshot_id: str
    constructed_at: datetime
    config_path: Path | None
    agent_config: AgentConfig
    workspace_root: Path
    source_documents: tuple[SourceDocument, ...]
    dependencies: tuple[ConfigurationDependency, ...]
    source_completeness: Literal["complete", "partial", "effective_only"]
    storage: LifecycleStorageDescription
    tool_catalog: tuple[ToolDefinition, ...]
    skill_catalog: tuple[SkillMetadata, ...]
    prepared: bool


@dataclass(frozen=True, slots=True)
class ConfigurationApplied:
    """本次真实 activation 采用的完整 Runner 配置，可直接类型化序列化。"""

    configuration_snapshot_id: str
    run_id: str
    session_id: str
    activation_id: str
    agent_id: str
    configuration: EffectiveConfiguration


def snapshot_tool_catalog(view: ToolRegistryView) -> tuple[ToolDefinition, ...]:
    """冻结实际可用目录，保留 eager/deferred 与 group 声明。"""
    return tuple(deepcopy(tool.definition) for tool in view.available_tools)


__all__ = [
    "ConfigurationApplied",
    "ConfigurationDependency",
    "LifecycleStorageDescription",
    "EffectiveConfiguration",
]
