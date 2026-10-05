"""候选配置和文件加载共用声明解析，保留文件基准及未修改 raw 数据。"""

from copy import deepcopy
from pathlib import Path

import pytest

from iris.agents import load_agent_config, parse_agent_config
from iris.exceptions import IrisConfigError


def test_raw_candidate_uses_original_path_without_rewriting_declarations(tmp_path: Path) -> None:
    """候选在写盘前解析，相对路径仍以原主 YAML 为基准。"""
    path = tmp_path / "configs" / "agent.yaml"
    raw = {"name": "test", "model": "openai/test", "context": {"path": "context.yaml"}}
    original = deepcopy(raw)
    parsed = parse_agent_config(raw, config_path=path)
    assert parsed.context.path == path.parent / "context.yaml"
    assert raw == original and not path.exists()


def test_file_and_candidate_share_raw_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """文件入口直接将 YAML 解析结果交给唯一模型解析器。"""
    from iris.agents.config import base

    path = tmp_path / "agent.yaml"
    path.write_text("name: test\nmodel: openai/test\nsystem: help\n", encoding="utf-8")
    original = base.parse_agent_config
    calls = []

    def observe(raw_config: dict[str, object], *, config_path: Path) -> base.AgentConfig:
        calls.append((raw_config, config_path))
        return original(raw_config, config_path=config_path)

    monkeypatch.setattr(base, "parse_agent_config", observe)
    assert load_agent_config(path).name == "test"
    assert len(calls) == 1 and calls[0][1] == path


def test_candidate_uses_existing_cross_field_invariants(tmp_path: Path) -> None:
    """修改叶字段不能绕过现有能力依赖，也不自动更改其它字段。"""
    raw = {
        "name": "test",
        "model": "openai/test",
        "system": "help",
        "context_policy": {"enabled": False},
        "todo": {"enabled": True},
    }
    with pytest.raises(IrisConfigError, match="context_policy"):
        parse_agent_config(raw, config_path=tmp_path / "agent.yaml")
    assert raw["context_policy"] == {"enabled": False}
