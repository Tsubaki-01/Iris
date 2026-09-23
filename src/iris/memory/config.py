"""记忆系统 config-first 声明面。

本模块把已解析的配置转换为 memory SDK 对象；AgentConfig 复用这些声明，
runtime 在确定 effective workspace 后调用服务构造入口。

Example:
    config = MemoryConfig(enabled=True)
    service = build_memory_service_from_config(config, workspace_root)
"""

# region imports
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..exceptions import IrisConfigError
from ..providers.protocols import CompletionProvider
from .generation_models import MemoryGenerationConfig
from .mirror import FileMemoryMirror
from .models import MemoryOverviewConfig
from .service import MemoryIOExecutionMode, MemoryService
from .sqlite import SQLiteMemoryStore

# endregion


class MemoryConfig(BaseModel):
    """记忆系统声明式配置。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    root: str = ".iris/memory"
    path: str = ".iris/memory/memory.db"
    read_namespaces: list[Annotated[str, StringConstraints(pattern=r"\S")]] = Field(
        default_factory=lambda: ["project"]
    )
    write_namespace: str = Field(default="project", pattern=r"\S")
    overview: MemoryOverviewConfig = Field(default_factory=MemoryOverviewConfig)
    generation: MemoryGenerationConfig = Field(default_factory=MemoryGenerationConfig)


def build_memory_service_from_config(
    config: MemoryConfig,
    workspace_root: Path,
    *,
    memory_service: MemoryService | None = None,
    overview_provider: CompletionProvider | None = None,
    overview_model: str | None = None,
) -> MemoryService | None:
    """按 memory 开关选择宿主服务或构造 SQLite 服务。

    Args:
        config: 已解析的 memory 配置。
        workspace_root: 调用方提供的 workspace 根目录。
        memory_service: 可选宿主服务，开启时原样复用，关闭时不挂载或操作。
        overview_provider: 宿主显式生成概览时使用的模型调用边界。
        overview_model: 显式概览请求使用的模型名称。

    Returns:
        MemoryService | None: 关闭返回 None；开启时复用注入对象或创建 SQLite 服务。

    Raises:
        IrisConfigError: 构造 SQLite 服务时 root/path 越界。
    """
    if not config.enabled:
        return None
    if memory_service is not None:
        return memory_service
    root = resolve_memory_path(config.root, workspace_root)
    path = resolve_memory_path(config.path, workspace_root)
    store = SQLiteMemoryStore(path)
    mirror = FileMemoryMirror(root, workspace_root=workspace_root)
    mirror.initialize_layout()
    return MemoryService(
        store,
        mirror=mirror,
        overview_provider=overview_provider,
        overview_model=overview_model,
        overview_config=config.overview,
        generation_provider=overview_provider,
        generation_model=overview_model,
        generation_config=config.generation,
        io_execution_mode=MemoryIOExecutionMode.THREAD,
    )


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
