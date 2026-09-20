"""记忆配置只接受概览与 namespace 的当前合同。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from iris.memory import MemoryConfig


@pytest.mark.parametrize(
    "config",
    [
        {"mirror": {"enabled": False}},
        {"recall_mode": "on_turn"},
        {"max_query_terms": None},
        {"max_query_terms": 128},
        {"write_policy": {"mode": "sdk_only"}},
        {"orchestrator": {"enabled": True}},
        {"scope": {"collection": "default"}},
        {"search": {"use_fts": True}},
    ],
)
def test_memory_config_rejects_settings_without_behavior(config: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MemoryConfig.model_validate(config)


def test_memory_defaults_share_project() -> None:
    config = MemoryConfig()
    assert config.read_namespaces == ["project"]
    assert config.write_namespace == "project"
    assert config.path == ".iris/memory/memory.db"


def test_memory_can_choose_separate_read_and_write_namespaces() -> None:
    config = MemoryConfig(
        read_namespaces=["project", "research/private"],
        write_namespace="research/private",
    )
    assert config.read_namespaces == ["project", "research/private"]
    assert config.write_namespace == "research/private"


def test_memory_overview_defaults_keep_generation_and_window_budgets_separate() -> None:
    """生成输入、生成输出和主请求采用预算分别配置。"""
    config = MemoryConfig()
    assert config.overview.input_budget_tokens == 96000
    assert config.overview.max_tokens == 1024
    assert config.overview.system_budget_ratio == 0.02
