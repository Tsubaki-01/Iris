"""独立 Decision 配置和 Agent 相对文件引用。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from iris.agents import AgentConfig, load_agent_config
from iris.decision.config import DecisionConfig, load_decision_config
from iris.exceptions import IrisConfigError


def test_agent_yaml_resolves_decision_relative_to_declaring_file(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: test\nmodel: openai/test\nsystem: instructions\n"
        "decision:\n  path: decisions/tools.yaml\n",
        encoding="utf-8",
    )
    config = load_agent_config(path)
    assert config.decision is not None
    assert config.decision.path == tmp_path / "decisions" / "tools.yaml"
    direct = AgentConfig.model_validate(
        {
            "name": "test",
            "model": "openai/test",
            "system": "instructions",
            "decision": {"path": "decision.yaml"},
        }
    )
    assert direct.decision is not None
    assert direct.decision.path == Path("decision.yaml")


def test_decision_defaults_and_implemented_switch(tmp_path: Path) -> None:
    path = tmp_path / "decision.yaml"
    path.write_text("tools:\n  discovery: true\n", encoding="utf-8")
    config = load_decision_config(path)
    assert config.provider == "typesafe"
    assert config.model == "jev-1.13.0"
    assert config.timeout_seconds == 5
    assert config.enabled
    assert not DecisionConfig().enabled
    with pytest.raises(ValidationError):
        config.tools.discovery = False


@pytest.mark.parametrize(
    "content",
    [
        "tools: {discovery: 'yes'}",
        "tools: {discovery: false, missing: true}",
        "unknown: 1",
        "provider: other",
        "timeout_seconds: 0",
        "model: '  '",
        "tools: [",
        "[]",
    ],
)
def test_invalid_decision_file_fails_at_config_boundary(tmp_path: Path, content: str) -> None:
    path = tmp_path / "decision.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(IrisConfigError):
        load_decision_config(path)


def test_missing_decision_file_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(IrisConfigError):
        load_decision_config(tmp_path / "missing.yaml")
