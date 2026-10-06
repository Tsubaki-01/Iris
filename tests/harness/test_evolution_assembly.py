"""宿主显式项目学习装配只使用选定主配置和实际依赖。"""

from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from iris.agents import AgentConfig
from iris.exceptions import IrisConfigError
from iris.harness.evolution import build_project_evolution_binding
from iris.message import LLMRequest, LLMResponse, Msg
from iris.observability import AgentObservabilityConfig, ObservabilityExportConfig
from iris.observability.service import Observability
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


@pytest.mark.asyncio
async def test_project_binding_wraps_raw_provider_once_and_borrows_service(tmp_path: Path) -> None:
    """绑定工厂不先包装 provider，服务不接管宿主的 exporter。"""
    sdk = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    sdk.add_span_processor(SimpleSpanProcessor(exporter))
    observability = Observability.from_config(
        AgentObservabilityConfig(enabled=True), ObservabilityExportConfig(), tracer_provider=sdk
    )
    config = AgentConfig.model_validate(
        {
            "name": "learner",
            "model": "openai/test",
            "system": "help",
            "skills": {"enabled": True},
            "evolution": {"enabled": True},
        }
    )
    raw = StaticProvider(LLMResponse(provider="test"))
    binding = build_project_evolution_binding(
        config,
        workspace_root=tmp_path,
        prompt_source=PromptSource.initialize(tmp_path),
        provider=raw,
        observability=observability,
    )
    assert binding is not None
    assert binding.service.observability is observability
    request = LLMRequest(model="test", messages=[Msg.user("inspect")])
    try:
        await binding.service.provider.complete(request)
        assert raw.requests == [request]
        assert len(exporter.get_finished_spans()) == 1
    finally:
        sdk.shutdown()


def test_config_targets_require_explicit_primary_path(tmp_path: Path) -> None:
    """内存构造的配置没有主文件时，不能猜测要修改哪一份 YAML。"""
    config = AgentConfig.model_validate(
        {
            "name": "learner",
            "model": "openai/test",
            "system": "help",
            "skills": {"enabled": True},
            "evolution": {"enabled": True, "config_targets": ["system"]},
        }
    )
    with pytest.raises(IrisConfigError, match="config_path"):
        build_project_evolution_binding(
            config,
            workspace_root=tmp_path,
            prompt_source=PromptSource.initialize(tmp_path),
            provider=StaticProvider(),
        )


def test_bindings_expose_only_open_targets_and_use_domain_contracts(tmp_path: Path) -> None:
    """候选收到同源说明，配置解析沿原文件基准，绑定不提供通用文件入口。"""
    config = AgentConfig.model_validate(
        {
            "name": "learner",
            "model": "openai/test",
            "system": "help",
            "skills": {"enabled": True},
            "evolution": {
                "enabled": True,
                "config_targets": ["compaction.input_budget_tokens"],
                "prompt_targets": ["memory_flush", "compaction_input"],
            },
        }
    )
    path = tmp_path / "configs" / "agent.yaml"
    binding = build_project_evolution_binding(
        config,
        workspace_root=tmp_path,
        prompt_source=PromptSource.initialize(tmp_path),
        provider=StaticProvider(),
        config_path=path,
    )
    assert {target.name for target in binding.service.prompt_targets} == {
        "memory_flush",
        "compaction_input",
    }
    target = binding.service.config_target
    assert target.path == path
    assert set(target.descriptions) == {"compaction.input_budget_tokens"}
    assert '"exclusiveMinimum": 0' in target.descriptions["compaction.input_budget_tokens"]
    raw = {"name": "test", "model": "openai/test", "context": {"path": "context.yaml"}}
    target.validate(raw)
    assert raw["context"]["path"] == "context.yaml" and not path.exists()
