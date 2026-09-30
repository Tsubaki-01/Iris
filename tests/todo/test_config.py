"""Todo 声明只在配置边界校验，不创建文件或自动注册工具。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from iris.agents import AgentConfig, build_tool_registry, load_agent_config
from iris.exceptions import IrisConfigError


def test_todo_is_disabled_by_default() -> None:
    """普通 Agent 不会自动维护待办文件。"""
    config = AgentConfig(name="agent", model="openai/test", system="system")
    assert not config.todo.enabled


def test_yaml_enables_todo_without_files_or_tools(tmp_path: Path) -> None:
    """加载声明不执行文件 IO，工具仍由独立声明决定。"""
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: agent\nmodel: openai/test\nsystem: system\ntodo:\n  enabled: true\n",
        encoding="utf-8",
    )
    config = load_agent_config(path)
    assert config.todo.enabled
    assert build_tool_registry(config.tools).view().available_tools == []
    assert not (tmp_path / ".iris").exists()


def test_enabled_todo_requires_context_policy() -> None:
    """每步投影要求 context policy，由 AgentConfig 统一处理。"""
    with pytest.raises(ValidationError, match="todo.enabled.*context_policy"):
        AgentConfig.model_validate(
            {
                "name": "agent",
                "model": "openai/test",
                "system": "system",
                "todo": {"enabled": True},
                "context_policy": {"enabled": False},
            }
        )


def test_yaml_wraps_todo_configuration_error(tmp_path: Path) -> None:
    """YAML 解析入口沿用项目配置错误，而非泄漏模型异常。"""
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: agent\nmodel: openai/test\nsystem: system\n"
        "todo:\n  enabled: true\ncontext_policy:\n  enabled: false\n",
        encoding="utf-8",
    )
    with pytest.raises(IrisConfigError, match="todo.enabled"):
        load_agent_config(path)
