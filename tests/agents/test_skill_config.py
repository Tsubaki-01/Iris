from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from iris.agents import AgentConfig, AgentSkillsConfig, load_agent_config
from iris.exceptions import IrisConfigError


def test_agent_skills_config_defaults_and_agent_shape_are_exact() -> None:
    skills = AgentSkillsConfig()
    config = AgentConfig(
        name="example",
        model="openai/gpt-4o-mini",
        system="Base instructions",
    )

    assert skills.enabled is False
    assert skills.root == ".agents/skills"
    assert skills.require == ()
    assert config.skills is None
    assert config.model_dump(mode="json")["skills"] is None


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


@pytest.mark.parametrize("root", ("", "   "))
def test_agent_skills_root_must_not_be_blank(root: str) -> None:
    with pytest.raises(ValidationError, match="root"):
        AgentSkillsConfig(root=root)


@pytest.mark.parametrize(
    "name",
    ("Bad_Name", "bad name", "-bad", "bad-", ""),
)
def test_agent_skills_require_uses_bare_kebab_case_names(name: str) -> None:
    with pytest.raises(ValidationError, match="require"):
        AgentSkillsConfig(require=(name,))


def test_agent_skills_config_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentSkillsConfig.model_validate({"enabled": True, "unknown": True})


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
