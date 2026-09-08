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
    ],
)
def test_memory_config_rejects_settings_without_behavior(config: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MemoryConfig.model_validate(config)
