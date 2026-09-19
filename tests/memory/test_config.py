from __future__ import annotations

import pytest
from pydantic import ValidationError

from iris.memory import MemoryConfig


@pytest.mark.parametrize(
    "config",
    [
        {"mirror": {"mode": "minimal"}},
        {"write_policy": {"mode": "sdk_only"}},
        {"orchestrator": {"enabled": True}},
        {"scope": {"collection": "default"}},
        {"search": {"use_fts": True}},
    ],
)
def test_memory_config_rejects_settings_without_behavior(config: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MemoryConfig.model_validate(config)


def test_memory_defaults_share_project_and_keep_query_budget_optional() -> None:
    config = MemoryConfig()
    assert config.read_namespaces == ["project"]
    assert config.write_namespace == "project"
    assert config.recall_mode == "on_turn"
    assert config.max_query_terms is None
    assert config.path == ".iris/memory/memory.db"


@pytest.mark.parametrize("budget", [0, -1])
def test_memory_query_budget_requires_a_positive_count(budget: int) -> None:
    with pytest.raises(ValidationError):
        MemoryConfig(max_query_terms=budget)


def test_memory_can_choose_manual_recall_and_separate_namespaces() -> None:
    config = MemoryConfig(
        recall_mode="manual",
        read_namespaces=["project", "research/private"],
        write_namespace="research/private",
        max_query_terms=128,
    )
    assert config.recall_mode == "manual"
    assert config.read_namespaces == ["project", "research/private"]
    assert config.write_namespace == "research/private"
