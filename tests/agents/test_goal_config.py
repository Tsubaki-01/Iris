"""Goal 配置的唯一外部校验边界。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from iris.agents import AgentConfig, load_agent_config
from iris.exceptions import IrisConfigError
from iris.goal import GoalConfig


def test_goal_defaults_are_disabled_without_changing_context_policy() -> None:
    """旧有普通 Agent 不会自动开启跨 Run 推进。"""
    config = AgentConfig(name="agent", model="openai/test", system="system")
    assert config.goal == GoalConfig()
    assert not config.goal.enabled
    assert config.goal.max_rounds == 20
    assert config.context_policy.enabled


def test_goal_yaml_preserves_explicit_budget(tmp_path: Path) -> None:
    """YAML 只解析 Goal 声明，不创建状态或存储。"""
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: agent\nmodel: openai/test\nsystem: system\n"
        "goal:\n  enabled: true\n  max_rounds: 3\n",
        encoding="utf-8",
    )
    config = load_agent_config(path)
    assert config.goal == GoalConfig(enabled=True, max_rounds=3)
    assert not (tmp_path / ".iris").exists()


def test_enabled_goal_requires_context_policy() -> None:
    """动态目标投影需要在配置解析处声明上下文依赖。"""
    with pytest.raises(ValidationError, match="context_policy"):
        AgentConfig.model_validate(
            {
                "name": "agent",
                "model": "openai/test",
                "system": "system",
                "goal": {"enabled": True},
                "context_policy": {"enabled": False},
            }
        )


@pytest.mark.parametrize("rounds", [0, -1])
def test_invalid_goal_round_limit_is_yaml_config_error(tmp_path: Path, rounds: int) -> None:
    """正数轮数约束由 GoalConfig 统一校验并映射配置错误。"""
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: agent\nmodel: openai/test\nsystem: system\n"
        f"goal:\n  enabled: true\n  max_rounds: {rounds}\n",
        encoding="utf-8",
    )
    with pytest.raises(IrisConfigError, match="配置校验失败"):
        load_agent_config(path)
