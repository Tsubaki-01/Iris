from pathlib import Path

import pytest

from examples.lifecycle import tools
from iris.agents import build_tool_registry, load_agent_config
from iris.context import ContextBuilder, load_context_build_input
from iris.exceptions import IrisToolValidationError

ROOT = Path(__file__).resolve().parents[2]


def test_chat_example_config_resolves_context_and_workspace() -> None:
    config = load_agent_config(ROOT / "examples/chat/agent.yaml")
    assert config.context is not None
    context = load_context_build_input(config.context.path)
    rendered = ContextBuilder().build(context)
    assert "Iris Example Agent" in rendered.system.text
    assert config.permissions.workspace == "workspace"
    assert config.session.backend == "sqlite"


def test_lifecycle_example_registers_wait_tool() -> None:
    config = load_agent_config(ROOT / "examples/lifecycle/agent.yaml")
    registry = build_tool_registry(config.tools)
    assert registry.get("wait_for_seconds").definition.name == "wait_for_seconds"
    assert config.session.path == ".iris/lifecycle.db"


def test_lifecycle_wait_tool_keeps_sync_smoke_semantics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    slept: list[int] = []
    monkeypatch.setattr(tools.time, "sleep", slept.append)

    assert tools.wait_for_seconds(2) == "waited 2 seconds"
    captured = capsys.readouterr()

    assert slept == [2]
    assert captured.out == ""
    assert captured.err == "IRIS_EXAMPLE_TOOL_STARTED seconds=2\n"


@pytest.mark.parametrize("seconds", [True, False, 0, 61, 1.5])
def test_lifecycle_wait_tool_rejects_invalid_seconds(seconds: int | float | bool) -> None:
    with pytest.raises(IrisToolValidationError):
        tools.wait_for_seconds(seconds)
