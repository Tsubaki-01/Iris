"""项目经验学习与显式开放的有限修订配置。"""

import pytest
from pydantic import ValidationError

from iris.evolution.config import EvolutionConfig


def test_defaults_and_explicit_revision_targets() -> None:
    config = EvolutionConfig()
    assert not config.enabled and config.policy_skill is None
    assert config.skill_max_chars == 8000
    assert (config.input_budget_tokens, config.output_budget_tokens) == (32000, 8000)
    assert config.prompt_targets == config.config_targets == ()
    configured = EvolutionConfig.model_validate(
        {"prompt_targets": ["compaction"], "config_targets": ["todo.enabled"]}
    )
    assert configured.prompt_targets == ("compaction",)
    assert configured.config_targets == ("todo.enabled",)


@pytest.mark.parametrize(
    "field", ["skill_max_chars", "input_budget_tokens", "output_budget_tokens"]
)
def test_budgets_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        EvolutionConfig.model_validate({field: 0})
