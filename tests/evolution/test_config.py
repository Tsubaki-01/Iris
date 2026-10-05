"""项目经验学习只接受 A 阶段的实际配置。"""

import pytest
from pydantic import ValidationError

from iris.evolution.config import EvolutionConfig


def test_defaults_and_no_future_revision_fields() -> None:
    config = EvolutionConfig()
    assert not config.enabled and config.policy_skill is None
    assert config.skill_max_chars == 8000
    assert (config.input_budget_tokens, config.output_budget_tokens) == (32000, 8000)
    with pytest.raises(ValidationError):
        EvolutionConfig.model_validate({"prompt_targets": ["compaction"]})


@pytest.mark.parametrize(
    "field", ["skill_max_chars", "input_budget_tokens", "output_budget_tokens"]
)
def test_budgets_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        EvolutionConfig.model_validate({field: 0})
