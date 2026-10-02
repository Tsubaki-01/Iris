from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from iris.agents import AgentConfig, load_agent_config
from iris.exceptions import IrisConfigError


def _raw() -> dict[str, object]:
    return {"name": "agent", "model": "openai/test", "system": "instructions"}


def test_hooks_default_empty_and_ordered_schema(tmp_path: Path) -> None:
    config = AgentConfig.model_validate(_raw())
    assert config.hooks == ()
    assert config.middleware.tools == ()

    raw = _raw() | {
        "hooks": [
            {
                "name": "first",
                "event": "tool.before",
                "tools": ["exec_command"],
                "handler": {"type": "command", "command": "python check.py"},
            },
            {
                "name": "first",
                "event": "run.finished",
                "timeout_seconds": 2.5,
                "handler": {
                    "type": "python",
                    "factory": "not_loaded:factory",
                    "options": {"arbitrary": {"nested": [1, True]}},
                },
            },
        ],
        "middleware": {"tools": [{"factory": "not_loaded:middleware", "options": {}}]},
    }
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_agent_config(path)

    assert [hook.event for hook in config.hooks] == ["tool.before", "run.finished"]
    assert [hook.name for hook in config.hooks] == ["first", "first"]
    assert config.hooks[0].tools == ("exec_command",)
    assert config.hooks[0].timeout_seconds == 10
    assert config.hooks[1].handler.options == {"arbitrary": {"nested": [1, True]}}
    assert config.middleware.tools[0].factory == "not_loaded:middleware"


@pytest.mark.parametrize(
    "patch",
    [
        {"event": "model.before"},
        {"name": "  "},
        {"tools": []},
        {"tools": [""]},
        {"event": "run.started", "tools": ["exec_command"]},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"handler": {"type": "future", "command": "echo hello"}},
        {"handler": {"type": "command", "command": " "}},
        {"handler": {"type": "command", "command": "echo hello", "options": {}}},
        {"handler": {"type": "python", "factory": "module:factory", "command": "echo"}},
        {"handler": {"type": "python", "factory": "module:factory", "options": []}},
        {"priority": 1},
    ],
)
def test_invalid_hooks_fail_at_yaml_boundary(tmp_path: Path, patch: dict[str, object]) -> None:
    hook = {
        "name": "hook",
        "event": "tool.before",
        "handler": {"type": "python", "factory": "module:factory"},
    } | patch
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(_raw() | {"hooks": [hook]}), encoding="utf-8")

    with pytest.raises(IrisConfigError, match="配置校验失败"):
        load_agent_config(path)


def test_middleware_rejects_unknown_levels_and_handler_options(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        yaml.safe_dump(_raw() | {"middleware": {"model": [{"factory": "module:factory"}]}}),
        encoding="utf-8",
    )
    with pytest.raises(IrisConfigError, match="配置校验失败"):
        load_agent_config(path)
