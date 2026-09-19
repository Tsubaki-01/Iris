"""Agent YAML 复用 MemoryConfig，不在加载时装配数据库。"""

from pathlib import Path

import pytest

from iris.agents import AgentConfig, load_agent_config
from iris.exceptions import IrisConfigError
from iris.memory import MemoryConfig


def test_agent_memory_is_disabled_by_default() -> None:
    config = AgentConfig.model_validate({"name": "a", "model": "openai/test", "system": "a"})
    assert isinstance(config.memory, MemoryConfig)
    assert config.memory.backend == "none"
    assert config.memory.read_namespaces == ["project"]


def test_agent_yaml_loads_memory_options_without_creating_files(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: a\nmodel: openai/test\nsystem: a\nmemory:\n"
        "  backend: sqlite\n  recall_mode: manual\n  max_query_terms: 128\n"
        "  read_namespaces: [project, research]\n  write_namespace: research\n",
        encoding="utf-8",
    )
    config = load_agent_config(path)
    assert config.memory.recall_mode == "manual"
    assert config.memory.max_query_terms == 128
    assert config.memory.read_namespaces == ["project", "research"]
    assert config.memory.write_namespace == "research"
    assert not (tmp_path / ".iris").exists()


@pytest.mark.parametrize("invalid", ["scope: {}", "max_query_terms: 0", "recall_mode: always"])
def test_agent_yaml_rejects_invalid_memory_config(tmp_path: Path, invalid: str) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        f"name: a\nmodel: openai/test\nsystem: a\nmemory:\n  {invalid}\n", encoding="utf-8"
    )
    with pytest.raises(IrisConfigError, match="Agent 配置校验失败"):
        load_agent_config(path)
