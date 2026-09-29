"""Agent 命令环境与独立执行权限的配置边界。"""

import pytest
from pydantic import ValidationError

from iris.agents import AgentConfig, DockerConfig, ExecutionConfig, PermissionsConfig
from iris.execution.models import ExecutionMode


def test_agent_execution_defaults_without_implicit_tool() -> None:
    """默认 Native 和 execute confirm 不会隐式注册命令入口。"""
    config = AgentConfig(name="agent", model="openai/model", system="system")
    assert config.execution == ExecutionConfig()
    assert config.execution.mode is ExecutionMode.NATIVE
    assert config.permissions.execute == "confirm"
    assert "exec.command" not in config.tools.builtin
    assert "execution" not in config.model_fields_set


def test_explicit_docker_config_preserves_field_source() -> None:
    """child 装配可以根据来源区分默认配置与显式 override。"""
    config = AgentConfig.model_validate(
        {
            "name": "agent",
            "model": "openai/model",
            "system": "system",
            "execution": {"mode": "docker"},
        }
    )
    assert config.execution.docker == DockerConfig()
    assert "execution" in config.model_fields_set


def test_native_rejects_docker_block_at_config_boundary() -> None:
    with pytest.raises(ValidationError):
        ExecutionConfig(mode="native", docker=DockerConfig())


@pytest.mark.parametrize("value", ["allow", "confirm", "deny"])
def test_execute_permission_is_independent_of_writes(value: str) -> None:
    config = PermissionsConfig(execute=value, writes="deny")
    assert config.execute == value
    assert config.writes == "deny"
