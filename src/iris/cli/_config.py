"""CLI 入口共享的 Agent 配置加载。"""

from __future__ import annotations

from pathlib import Path

from ..agents import AgentConfig, load_agent_config
from ..config import init_config, is_config_initialized


def load_cli_agent(
    config_path: Path,
    *,
    env_file: Path | None = None,
) -> AgentConfig:
    """初始化可选 dotenv 配置并加载一份 Agent YAML。"""
    if not is_config_initialized():
        init_config(env_file=str(env_file) if env_file is not None else None)
    return load_agent_config(config_path)


__all__ = ["load_cli_agent"]
