"""宿主显式项目学习装配只使用选定主配置和实际依赖。"""

from pathlib import Path

from iris.agents import AgentConfig
from iris.harness.evolution import build_project_evolution_binding
from iris.prompts import PromptSource

from .fakes import StaticProvider


def test_disabled_evolution_does_not_create_material_store(tmp_path: Path) -> None:
    """默认配置没有隐藏的项目学习资源。"""
    config = AgentConfig(name="plain", model="openai/test", system="help")
    assert (
        build_project_evolution_binding(
            config,
            workspace_root=tmp_path,
            prompt_source=PromptSource.initialize(tmp_path),
            provider=StaticProvider(),
        )
        is None
    )
    assert not (tmp_path / ".iris" / "evolution").exists()


def test_project_binding_uses_explicit_host_resources(tmp_path: Path) -> None:
    """Memory 关闭时仍可装配，Skill 路径沿用项目根解析规则。"""
    config = AgentConfig.model_validate(
        {
            "name": "learner",
            "model": "openai/test",
            "system": "使用项目经验",
            "skills": {"enabled": True, "root": "knowledge/skills"},
            "evolution": {"enabled": True},
            "memory": {"enabled": False},
        }
    )
    source = PromptSource.initialize(tmp_path, "custom-prompts")
    provider = StaticProvider()
    binding = build_project_evolution_binding(
        config, workspace_root=tmp_path, prompt_source=source, provider=provider
    )
    assert binding.workspace_root == tmp_path
    assert binding.service.prompt_source is source
    assert binding.service.provider is provider
    assert binding.service.config is config.evolution
    assert binding.service.skill_path == tmp_path / "knowledge/skills/project-experience/SKILL.md"
    assert not (tmp_path / ".iris" / "memory").exists()
