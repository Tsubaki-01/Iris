"""Sub Agent child provider 的 SDK 注入协议。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..agents import AgentConfig
from ..runtime.environment import RuntimeProvider


class ChildProviderFactory(Protocol):
    """按选中 child 的普通配置构造独立 provider。"""

    def __call__(self, config: AgentConfig, *, config_path: Path) -> RuntimeProvider:
        """接收已加载的 child 配置和声明路径。"""
        ...
