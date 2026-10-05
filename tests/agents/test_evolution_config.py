"""项目学习声明只约束需要的 Skill 能力，不依赖 Memory。"""

import pytest
from pydantic import ValidationError

from iris.agents import AgentConfig


def test_evolution_defaults_disabled() -> None:
    """普通 Agent 不主动建立项目学习能力。"""
    config = AgentConfig(name="plain", model="openai/test", system="简洁回答")
    assert not config.evolution.enabled


@pytest.mark.parametrize("skills", [None, {"enabled": False}])
def test_evolution_requires_skill_discovery(skills: dict[str, bool] | None) -> None:
    """生成的项目经验必须有实际的 Skill 消费入口。"""
    with pytest.raises(ValidationError, match="skills.enabled"):
        AgentConfig.model_validate(
            {
                "name": "learner",
                "model": "openai/test",
                "system": "使用项目经验",
                "skills": skills,
                "evolution": {"enabled": True},
            }
        )


def test_evolution_does_not_require_memory() -> None:
    """关闭 Memory 时仍允许项目 Skill 学习。"""
    config = AgentConfig.model_validate(
        {
            "name": "learner",
            "model": "openai/test",
            "system": "使用项目经验",
            "skills": {"enabled": True},
            "evolution": {"enabled": True},
            "memory": {"enabled": False},
        }
    )
    assert config.evolution.enabled and not config.memory.enabled
