"""有限 config 字段赋值由原声明解析入口验证一次。"""

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from iris.agents.config.base import parse_agent_config
from iris.evolution.config import EvolutionConfig
from iris.evolution.models import RevisionTarget
from iris.evolution.revision import (
    ConfigTarget,
    prepare_candidate,
    prepare_revision,
    publish_revision,
)
from iris.exceptions import IrisEvolutionError
from iris.prompts import PromptSource

from .test_prompt_revision import response


def test_valid_candidate_preserves_raw_relative_paths_and_unmodified_values(tmp_path: Path) -> None:
    path = tmp_path / "config" / "agent.yaml"
    path.parent.mkdir()
    raw = {
        "name": "project",
        "model": "openai/test",
        "system": "遵守项目约定",
        "permissions": {"workspace": "../workspace"},
        "tools": {"builtin": ["file.read"]},
        "compaction": {"input_budget_tokens": 10000},
    }
    baseline = yaml.safe_dump(raw, allow_unicode=True)
    path.write_text(baseline, encoding="utf-8")
    validated: list[dict[str, Any]] = []

    def validate(candidate: dict[str, Any]) -> None:
        validated.append(copy.deepcopy(candidate))
        parsed = parse_agent_config(candidate, config_path=path)
        assert parsed.compaction.input_budget_tokens == 20000

    context = prepare_revision(
        prompt_snapshot=PromptSource.initialize(tmp_path).snapshot(),
        targets=(RevisionTarget(kind="config", name="compaction.input_budget_tokens"),),
        prompt_targets=(),
        config_target=ConfigTarget(
            path, validate, {"compaction.input_budget_tokens": "压缩输入预算"}
        ),
    )
    candidate = prepare_candidate(
        response(
            {
                "action": "config",
                "assignments": {"compaction.input_budget_tokens": 20000},
                "reason": "调整预算",
            }
        ),
        context,
    )
    assert len(validated) == 1
    assert path.read_text(encoding="utf-8") == baseline
    assert publish_revision(candidate)
    expected = copy.deepcopy(raw)
    expected["compaction"]["input_budget_tokens"] = 20000
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == expected


def test_assignment_does_not_change_an_unrequested_yaml_alias(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: project\nmodel: openai/test\nsystem: 规则\n"
        "compaction: &shared_budget\n  input_budget_tokens: 96000\n"
        "memory:\n  overview: *shared_budget\n",
        encoding="utf-8",
    )
    parse_agent_config(yaml.safe_load(path.read_text(encoding="utf-8")), config_path=path)
    context = prepare_revision(
        prompt_snapshot=PromptSource.initialize(tmp_path).snapshot(),
        targets=(RevisionTarget(kind="config", name="compaction.input_budget_tokens"),),
        prompt_targets=(),
        config_target=ConfigTarget(path, lambda raw: parse_agent_config(raw, config_path=path), {}),
    )
    candidate = prepare_candidate(
        response(
            {
                "action": "config",
                "assignments": {"compaction.input_budget_tokens": 20000},
                "reason": "只调整压缩预算",
            }
        ),
        context,
    )
    assert publish_revision(candidate)
    published = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert published["compaction"]["input_budget_tokens"] == 20000
    assert published["memory"]["overview"]["input_budget_tokens"] == 96000
    assert context.raw_config["compaction"]["input_budget_tokens"] == 96000


@pytest.mark.parametrize(
    "assignments",
    [
        {"compaction.input_budget_tokens": 0},
        {"todo.enabled": True},
        {"context_policy.enabled": True},
        {"model": "openai/other"},
        {"compaction": {"input_budget_tokens": 20000}},
    ],
)
def test_invalid_values_or_unopened_targets_do_not_change_other_fields(
    tmp_path: Path, assignments: dict[str, Any]
) -> None:
    path = tmp_path / "agent.yaml"
    raw = {
        "name": "project",
        "model": "openai/test",
        "system": "规则",
        "context_policy": {"enabled": False},
    }
    baseline = yaml.safe_dump(raw, allow_unicode=True)
    path.write_text(baseline, encoding="utf-8")
    context = prepare_revision(
        prompt_snapshot=PromptSource.initialize(tmp_path).snapshot(),
        targets=tuple(
            RevisionTarget(kind="config", name=name)
            for name in ("compaction.input_budget_tokens", "todo.enabled")
        ),
        prompt_targets=(),
        config_target=ConfigTarget(
            path, lambda value: parse_agent_config(value, config_path=path), {}
        ),
    )
    with pytest.raises(IrisEvolutionError):
        prepare_candidate(
            response({"action": "config", "assignments": assignments, "reason": "修改"}), context
        )
    assert path.read_text(encoding="utf-8") == baseline


def test_config_declarations_only_accept_supported_target_names() -> None:
    assert EvolutionConfig().prompt_targets == ()
    assert EvolutionConfig().config_targets == ()
    with pytest.raises(ValidationError):
        EvolutionConfig(config_targets=["context_policy.enabled"])
    with pytest.raises(ValidationError):
        EvolutionConfig(prompt_targets=["evolution_review"])


def test_system_target_cannot_switch_structured_context_or_remove_simple_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent.yaml"
    source = PromptSource.initialize(tmp_path)
    calls: list[dict[str, Any]] = []

    def validate(raw: dict[str, Any]) -> None:
        calls.append(raw)
        parse_agent_config(raw, config_path=path)

    for original, value in (
        ({"context": {"path": "context.yaml"}}, "新文本"),
        ({"system": "原文本"}, None),
    ):
        raw = {"name": "project", "model": "openai/test", **original}
        baseline = yaml.safe_dump(raw, allow_unicode=True)
        path.write_text(baseline, encoding="utf-8")
        context = prepare_revision(
            prompt_snapshot=source.snapshot(),
            targets=(RevisionTarget(kind="config", name="system"),),
            prompt_targets=(),
            config_target=ConfigTarget(path, validate, {"system": "简单模式系统指令"}),
        )
        with pytest.raises(IrisEvolutionError):
            prepare_candidate(
                response({"action": "config", "assignments": {"system": value}, "reason": "修改"}),
                context,
            )
        assert path.read_text(encoding="utf-8") == baseline
    assert len(calls) == 1


def test_config_publish_checks_baseline_and_model_input_marks_unknown_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text("name: project\nmodel: openai/test\nsystem: 原文\n", encoding="utf-8")
    context = prepare_revision(
        prompt_snapshot=PromptSource.initialize(tmp_path).snapshot(),
        targets=(RevisionTarget(kind="config", name="compaction.input_budget_tokens"),),
        prompt_targets=(),
        config_target=ConfigTarget(
            path,
            lambda raw: parse_agent_config(raw, config_path=path),
            {"compaction.input_budget_tokens": "输入预算"},
        ),
    )
    assert "未知" in context.model_input["historical_context"]
    assert context.model_input["targets"][0]["declared"] is False
    candidate = prepare_candidate(
        response(
            {
                "action": "config",
                "assignments": {"compaction.input_budget_tokens": 20000},
                "reason": "修改",
            }
        ),
        context,
    )
    path.write_text("用户已更新文件", encoding="utf-8")
    assert not publish_revision(candidate)
    assert path.read_text(encoding="utf-8") == "用户已更新文件"
