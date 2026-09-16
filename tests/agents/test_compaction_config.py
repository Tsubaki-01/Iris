"""自动压缩配置通过已有 YAML 和 SDK 声明路径生效。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import litellm
import pytest
from pydantic import ValidationError

from iris.agents import AgentConfig, CompactionConfig, load_agent_config
from iris.exceptions import IrisConfigError


def test_compaction_defaults_and_derived_budgets() -> None:
    config = AgentConfig.model_validate(
        {"name": "agent", "model": "openai/test", "system": "instructions"}
    )

    assert config.compaction == CompactionConfig()
    assert config.compaction.model_dump() == {
        "input_budget_tokens": 96000,
        "keep_recent_ratio": 0.15,
        "summary_ratio": 0.05,
        "timeout_seconds": 300,
        "prompt": None,
    }
    assert config.compaction.trigger_tokens == 76800
    assert config.compaction.keep_recent_tokens == 14400
    assert config.compaction.summary_tokens == 4800


def test_compaction_sdk_override_keeps_recent_target_soft() -> None:
    compaction = CompactionConfig(
        input_budget_tokens=101,
        keep_recent_ratio=0.7,
        summary_ratio=0.3,
        timeout_seconds=30.5,
    )
    config = AgentConfig.model_validate(
        {
            "name": "agent",
            "model": "openai/test",
            "system": "instructions",
            "compaction": compaction,
        }
    )

    assert config.compaction is compaction
    assert compaction.trigger_tokens == 80
    assert compaction.keep_recent_tokens == 71
    assert compaction.summary_tokens == 31


def test_load_compaction_config_without_model_or_tokenizer_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    count = Mock(side_effect=AssertionError("配置加载不能计量 token"))
    complete = Mock(side_effect=AssertionError("配置加载不能调用模型"))
    monkeypatch.setattr(litellm, "token_counter", count)
    monkeypatch.setattr(litellm, "acompletion", complete)
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: agent\nmodel: openai/test\nsystem: instructions\n"
        "compaction:\n  input_budget_tokens: 32000\n  keep_recent_ratio: 0.2\n"
        "  summary_ratio: 0.1\n  timeout_seconds: 120\n",
        encoding="utf-8",
    )

    config = load_agent_config(path)

    assert config.compaction == CompactionConfig(
        input_budget_tokens=32000,
        keep_recent_ratio=0.2,
        summary_ratio=0.1,
        timeout_seconds=120,
    )
    count.assert_not_called()
    complete.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_budget_tokens", 0),
        ("input_budget_tokens", -1),
        ("input_budget_tokens", 1.5),
        ("timeout_seconds", 0),
        ("timeout_seconds", -1),
        ("keep_recent_ratio", 0),
        ("keep_recent_ratio", 1),
        ("keep_recent_ratio", float("inf")),
        ("keep_recent_ratio", float("nan")),
        ("summary_ratio", 0),
        ("summary_ratio", 1),
        ("summary_ratio", float("-inf")),
        ("summary_ratio", float("nan")),
    ],
)
def test_compaction_rejects_invalid_budget_at_config_boundary(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=field):
        CompactionConfig.model_validate({field: value})


def test_yaml_invalid_compaction_is_wrapped_as_config_error(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: agent\nmodel: openai/test\nsystem: instructions\ncompaction:\n  summary_ratio: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(IrisConfigError) as caught:
        load_agent_config(path)

    assert "compaction.summary_ratio" in caught.value.context["error"]


def test_prompt_path_resolves_from_agent_yaml_without_reading_file(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: agent\nmodel: openai/test\nsystem: instructions\n"
        "compaction:\n  prompt: prompts/summary.j2\n",
        encoding="utf-8",
    )
    config = load_agent_config(path)
    assert config.compaction.prompt == (tmp_path / "prompts" / "summary.j2").resolve()
    assert not config.compaction.prompt.exists()
