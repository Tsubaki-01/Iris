"""记忆系统 config-first 声明面。

本模块只负责把简单配置转换为 Stage 1/2 的 memory SDK 对象，不解析 YAML 文件，
也不接入 agent runtime。

Example:
    config = MemoryConfig(backend="sqlite")
    service = build_memory_service_from_config(config, workspace_root)
"""

# region imports
from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import IrisConfigError
from .mirror import FileMemoryMirror
from .models import MemoryScope, MemoryVisibility
from .service import MemoryService
from .sqlite import SQLiteMemoryStore

# endregion


class MemoryBackend(StrEnum):
    """记忆持久化后端。"""

    NONE = "none"
    SQLITE = "sqlite"


class MemoryMirrorMode(StrEnum):
    """文件 mirror 模式。"""

    MINIMAL = "minimal"


class MemoryScopeConfig(BaseModel):
    """记忆 scope 的配置默认值。"""

    model_config = ConfigDict(extra="forbid")

    collection: str = "default"
    visibility: MemoryVisibility = MemoryVisibility.AGENT

    def to_scope(
        self,
        *,
        workspace_id: str,
        agent_id: str,
        session_id: str | None = None,
    ) -> MemoryScope:
        """结合运行时标识构造 `MemoryScope`。

        Args:
            workspace_id: 运行时 workspace 标识。
            agent_id: 运行时 agent 标识。
            session_id: 可选运行时 session 标识。仅 `visibility=session` 时写入 scope；
                agent/workspace 级记忆会忽略该值以保持跨会话可见。

        Returns:
            MemoryScope: 可传给 Stage 1 SDK 的 scope。

        Raises:
            IrisConfigError: 当配置的 visibility 与运行时 session 参数不匹配时抛出。
        """
        effective_session_id = (
            session_id if self.visibility == MemoryVisibility.SESSION else None
        )
        try:
            return MemoryScope(
                workspace_id=workspace_id,
                agent_id=agent_id,
                collection=self.collection,
                visibility=self.visibility,
                session_id=effective_session_id,
            )
        except ValueError as exc:
            raise IrisConfigError(
                "memory scope 配置无效",
                visibility=self.visibility.value,
                session_id=effective_session_id,
            ) from exc


class MemorySearchConfig(BaseModel):
    """记忆搜索配置。"""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=10, gt=0, le=100)
    use_fts: bool = True


class MemoryMirrorConfig(BaseModel):
    """记忆 mirror 配置。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    mode: MemoryMirrorMode = MemoryMirrorMode.MINIMAL


class MemoryWritePolicyConfig(BaseModel):
    """记忆写入策略配置。"""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["sdk_only"] = "sdk_only"
    delete_mode: Literal["tombstone"] = "tombstone"


class MemoryOrchestratorConfig(BaseModel):
    """记忆编排器配置占位。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class MemoryConfig(BaseModel):
    """记忆系统声明式配置。"""

    model_config = ConfigDict(extra="forbid")

    backend: MemoryBackend = MemoryBackend.NONE
    root: str = ".iris/memory"
    path: str = ".iris/memory/memory.db"
    scope: MemoryScopeConfig = Field(default_factory=MemoryScopeConfig)
    search: MemorySearchConfig = Field(default_factory=MemorySearchConfig)
    mirror: MemoryMirrorConfig = Field(default_factory=MemoryMirrorConfig)
    write_policy: MemoryWritePolicyConfig = Field(default_factory=MemoryWritePolicyConfig)
    orchestrator: MemoryOrchestratorConfig = Field(default_factory=MemoryOrchestratorConfig)


def build_memory_service_from_config(
    config: MemoryConfig,
    workspace_root: Path,
) -> MemoryService | None:
    """从 memory 配置构造服务。

    Args:
        config: 已解析的 memory 配置。
        workspace_root: 调用方提供的 workspace 根目录。

    Returns:
        MemoryService | None: `backend=none` 返回 None；SQLite 后端返回可用服务。

    Raises:
        IrisConfigError: 当 backend 或 root/path 越界时抛出。
    """
    if config.backend == MemoryBackend.NONE:
        return None
    if config.backend != MemoryBackend.SQLITE:
        raise IrisConfigError("不支持的 memory backend", backend=config.backend.value)

    root = resolve_memory_path(config.root, workspace_root)
    path = resolve_memory_path(config.path, workspace_root)
    store = SQLiteMemoryStore(path, use_fts=config.search.use_fts)
    mirror: FileMemoryMirror | None = None
    if config.mirror.enabled:
        mirror = FileMemoryMirror(root)
        mirror.initialize_layout()
    return MemoryService(store, mirror=mirror)


def resolve_memory_path(value: str, workspace_root: Path) -> Path:
    """将 memory 配置路径解析到 workspace 内。

    Args:
        value: 配置中声明的相对或绝对路径。
        workspace_root: workspace 根目录。

    Returns:
        Path: resolve 后的绝对路径。

    Raises:
        IrisConfigError: 当路径为空或逃逸 workspace 时抛出。
    """
    if not value.strip():
        raise IrisConfigError("memory 路径不能为空")
    root = workspace_root.resolve(strict=False)
    raw_path = Path(value)
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise IrisConfigError(
            "memory 路径不在 workspace 内",
            path=value,
            workspace_root=str(root),
        ) from exc
    return resolved
