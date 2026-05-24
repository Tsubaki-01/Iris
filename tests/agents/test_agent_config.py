from __future__ import annotations

from pathlib import Path

import pytest

from iris.agents import AgentConfig, load_agent_config
from iris.core import ModelRoute
from iris.exceptions import IrisConfigError


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_agent_config_accepts_structured_model(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "agent.yaml",
        """
name: coding-agent
model:
  provider: openai
  name: gpt-4o-mini
system: 你是一个本地助手。
tools:
  builtin:
    - file.read
permissions:
  workspace: .
  writes: confirm
session:
  backend: sqlite
  path: .iris/session.db
""",
    )

    config = load_agent_config(config_path)

    assert isinstance(config, AgentConfig)
    assert config.name == "coding-agent"
    assert config.model.provider == "openai"
    assert config.model.name == "gpt-4o-mini"
    assert config.to_model_route() == ModelRoute(provider="openai", model="gpt-4o-mini")
    assert config.tools.builtin == ["file.read"]
    assert config.permissions.writes == "confirm"
    assert config.session.backend == "sqlite"


def test_load_agent_config_accepts_route_string_model(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "agent.yaml",
        """
name: route-agent
model: openai/gpt-4o-mini
system: 你是一个本地助手。
""",
    )

    config = load_agent_config(config_path)

    assert config.model.provider == "openai"
    assert config.model.name == "gpt-4o-mini"
    assert config.to_model_route() == ModelRoute(provider="openai", model="gpt-4o-mini")
    assert config.session.backend == "none"


def test_load_agent_config_defaults_sqlite_session_path(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "agent.yaml",
        """
name: sqlite-agent
model: openai/gpt-4o-mini
system: 你是一个本地助手。
session:
  backend: sqlite
""",
    )

    config = load_agent_config(config_path)

    assert config.session.backend == "sqlite"
    assert config.session.path == ".iris/session.db"


@pytest.mark.parametrize(
    "content",
    [
        """
model: openai/gpt-4o-mini
system: 你是一个本地助手。
""",
        """
name: bad-model
model: gpt-4o-mini
system: 你是一个本地助手。
""",
        """
name: bad-session
model: openai/gpt-4o-mini
system: 你是一个本地助手。
session:
  backend: redis
""",
        """
name: bad-writes
model: openai/gpt-4o-mini
system: 你是一个本地助手。
permissions:
  writes: maybe
""",
    ],
)
def test_load_agent_config_rejects_invalid_config(tmp_path: Path, content: str) -> None:
    config_path = _write_yaml(tmp_path / "agent.yaml", content)

    with pytest.raises(IrisConfigError):
        load_agent_config(config_path)


def test_load_agent_config_rejects_inline_python_script(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "agent.yaml",
        """
name: script-agent
model: openai/gpt-4o-mini
system: 你是一个本地助手。
tools:
  python:
    script: |
      def search_notes(query: str) -> str:
          return query
""",
    )

    with pytest.raises(IrisConfigError, match="script"):
        load_agent_config(config_path)


def test_load_agent_config_rejects_mixed_python_tool_list(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "agent.yaml",
        """
name: mixed-tools
model: openai/gpt-4o-mini
system: 你是一个本地助手。
tools:
  python:
    - my_project.tools:search_notes
""",
    )

    with pytest.raises(IrisConfigError, match="functions"):
        load_agent_config(config_path)


def test_load_agent_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IrisConfigError):
        load_agent_config(tmp_path / "missing.yaml")


def test_load_agent_config_wraps_unreadable_path_errors(tmp_path: Path) -> None:
    with pytest.raises(IrisConfigError):
        load_agent_config(tmp_path)
