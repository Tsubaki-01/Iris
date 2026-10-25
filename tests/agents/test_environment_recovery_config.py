"""环境恢复声明与 child 共享边界。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from iris.agents import AgentConfig, EnvironmentRecoveryConfig
from iris.exceptions import IrisConfigError
from iris.runtime._assembly import resolve_runtime_boundary


def test_default_and_explicit_environment_recovery(tmp_path: Path) -> None:
    """声明只保存开关，不访问资源。"""
    config = AgentConfig(name="test", model="openai/test", system="test")
    assert config.environment_recovery == EnvironmentRecoveryConfig(enabled=False)
    for enabled in (False, True):
        parsed = AgentConfig.model_validate(
            {
                "name": "test",
                "model": "openai/test",
                "system": "test",
                "environment_recovery": {"enabled": enabled},
            }
        )
        assert parsed.environment_recovery.enabled is enabled
    with pytest.raises(ValidationError):
        EnvironmentRecoveryConfig.model_validate({"directory": str(tmp_path)})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("enabled", [False, True])
def test_child_explicit_recovery_is_rejected(enabled: bool) -> None:
    """child 显式 false 也不能覆盖 root 的共享开关。"""
    root = AgentConfig(name="root", model="openai/test", system="test")
    boundary = resolve_runtime_boundary(root)
    child = AgentConfig.model_validate(
        {
            "name": "child",
            "model": "openai/test",
            "system": "test",
            "environment_recovery": {"enabled": enabled},
        }
    )
    with pytest.raises(IrisConfigError, match="environment_recovery"):
        resolve_runtime_boundary(child, parent_boundary=boundary)
    inherited = resolve_runtime_boundary(root, parent_boundary=boundary)
    assert inherited.command_binding is boundary.command_binding
