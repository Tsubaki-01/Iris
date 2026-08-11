from __future__ import annotations

from pathlib import Path

from fakes import FakeProvider

from iris.message import LLMResponse, TextBlock
from iris.runtime import AgentRuntime, RuntimeFactory


def _response() -> LLMResponse:
    return LLMResponse(provider="fake", content=[TextBlock(text="完成")])


def test_from_config_path_loads_relative_context_without_creating_store(
    tmp_path: Path,
) -> None:
    context_path = tmp_path / "context.yaml"
    context_path.write_text(
        "system:\n  slots:\n    - name: instructions\n      content: 来自 context.yaml\n",
        encoding="utf-8",
    )
    agent_path = tmp_path / "agent.yaml"
    agent_path.write_text(
        "\n".join(
            [
                "name: context-agent",
                "model: openai/gpt-4o-mini",
                "context:",
                "  path: context.yaml",
                "session:",
                "  backend: sqlite",
                "  path: state/lifecycle.db",
            ]
        ),
        encoding="utf-8",
    )

    runtime = RuntimeFactory.from_config_path(
        agent_path,
        provider=FakeProvider([_response()]),
    )

    assert isinstance(runtime, AgentRuntime)
    assert runtime.environment.context_input.system.slots[0].content == "来自 context.yaml"
    assert runtime.environment.agent_config.context is not None
    assert runtime.environment.agent_config.context.path == context_path.resolve()
    assert not (tmp_path / "state" / "lifecycle.db").exists()
