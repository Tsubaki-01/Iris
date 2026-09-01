from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from iris.agents import AgentSkillsConfig, load_agent_config
from iris.exceptions import IrisConfigError


def test_agent_skills_config_normalizes_root_and_preserves_require_order() -> None:
    skills = AgentSkillsConfig(
        enabled=True,
        root="  project-skills  ",
        require=("bravo", "alpha", "bravo"),
    )

    assert skills.root == "project-skills"
    assert skills.require == ("bravo", "alpha", "bravo")


@pytest.mark.parametrize("enabled", (1, 0, "true", "false"))
def test_agent_skills_enabled_requires_strict_boolean(enabled: object) -> None:
    with pytest.raises(ValidationError, match="enabled"):
        AgentSkillsConfig(enabled=enabled)  # type: ignore[arg-type]


def test_load_agent_config_wraps_unknown_skills_field(tmp_path: Path) -> None:
    agent_path = tmp_path / "agent.yaml"
    agent_path.write_text(
        "\n".join(
            [
                "name: example",
                "model: openai/gpt-4o-mini",
                "system: Base instructions",
                "skills:",
                "  enabled: true",
                "  unknown: value",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(IrisConfigError, match="Agent 配置校验失败") as exc_info:
        load_agent_config(agent_path)

    assert "skills.unknown" in exc_info.value.context["error"]
