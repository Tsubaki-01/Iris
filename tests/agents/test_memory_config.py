"""Agent YAML 复用 MemoryConfig，加载只解析当前合同。"""

from pathlib import Path

import pytest

from iris.agents import AgentConfig, load_agent_config
from iris.exceptions import IrisConfigError
from iris.memory import MemoryConfig


def test_agent_memory_is_disabled_by_default() -> None:
    config = AgentConfig.model_validate({"name": "a", "model": "openai/test", "system": "a"})
    assert isinstance(config.memory, MemoryConfig)
    assert config.memory.enabled is False
    assert config.memory.read_namespaces == ["project"]


def test_agent_yaml_loads_memory_options_without_creating_files(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: a\nmodel: openai/test\nsystem: a\nmemory:\n"
        "  enabled: true\n  read_namespaces: [project, research]\n"
        "  write_namespace: research\n",
        encoding="utf-8",
    )
    config = load_agent_config(path)
    assert config.memory.enabled is True
    assert config.memory.read_namespaces == ["project", "research"]
    assert config.memory.write_namespace == "research"
    assert not (tmp_path / ".iris").exists()


@pytest.mark.parametrize(
    "invalid",
    [
        "backend: none",
        "backend: sqlite",
        "scope: {}",
        "max_query_terms: 128",
        "recall_mode: manual",
        "mirror: {}",
    ],
)
def test_agent_yaml_rejects_invalid_memory_config(tmp_path: Path, invalid: str) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        f"name: a\nmodel: openai/test\nsystem: a\nmemory:\n  {invalid}\n", encoding="utf-8"
    )
    with pytest.raises(IrisConfigError, match="Agent 配置校验失败"):
        load_agent_config(path)


def test_agent_yaml_loads_overview_budgets_without_side_effects(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: a\nmodel: openai/test\nsystem: a\nmemory:\n"
        "  enabled: true\n  overview:\n"
        "    input_budget_tokens: 2048\n    max_tokens: 256\n    system_budget_ratio: 0.03\n",
        encoding="utf-8",
    )
    config = load_agent_config(path)
    assert config.memory.overview.input_budget_tokens == 2048
    assert config.memory.overview.max_tokens == 256
    assert config.memory.overview.system_budget_ratio == 0.03
    assert not (tmp_path / ".iris").exists()
