"""Agent 命令环境与独立执行权限的配置边界。"""

import pytest
from pydantic import ValidationError

from iris.agents import AgentConfig, CommandConfig, DockerConfig, PermissionsConfig
from iris.command.models import CommandMode


def test_agent_command_defaults_without_implicit_tool() -> None:
    """默认 Native 和 execute confirm 不会隐式注册命令入口。"""
    config = AgentConfig(name="agent", model="openai/model", system="system")
    assert config.command == CommandConfig()
    assert config.command.mode is CommandMode.NATIVE
    assert config.permissions.execute == "confirm"
    assert "exec.command" not in config.tools.builtin
    assert "command" not in config.model_fields_set


def test_explicit_docker_config_preserves_field_source() -> None:
    """child 装配可以根据来源区分默认配置与显式 override。"""
    config = AgentConfig.model_validate(
        {
            "name": "agent",
            "model": "openai/model",
            "system": "system",
            "command": {"mode": "docker"},
        }
    )
    assert config.command.docker == DockerConfig()
    assert "command" in config.model_fields_set
    assert config.model_dump(mode="json")["command"]["mode"] == "docker"


def test_native_rejects_docker_block_at_config_boundary() -> None:
    with pytest.raises(ValidationError):
        CommandConfig(mode="native", docker=DockerConfig())


@pytest.mark.parametrize("value", ["allow", "confirm", "deny"])
def test_execute_permission_is_independent_of_writes(value: str) -> None:
    config = PermissionsConfig(execute=value, writes="deny")
    assert config.execute == value
    assert config.writes == "deny"
