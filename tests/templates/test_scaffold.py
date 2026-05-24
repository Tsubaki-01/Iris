from __future__ import annotations

from pathlib import Path

import pytest

from iris.agents import build_tool_registry, load_agent_config
from iris.exceptions import IrisTemplateError, IrisTemplateNotFoundError
from iris.templates import scaffold_template


def test_scaffold_template_copies_file_agent(tmp_path: Path) -> None:
    written = scaffold_template("file-agent", tmp_path)

    assert tmp_path / "agent.yaml" in written
    assert tmp_path / "README.md" in written
    assert (tmp_path / "agent.yaml").is_file()
    assert (tmp_path / "README.md").is_file()


def test_scaffold_template_does_not_overwrite_existing_files(tmp_path: Path) -> None:
    (tmp_path / "agent.yaml").write_text("name: existing\n", encoding="utf-8")

    with pytest.raises(IrisTemplateError):
        scaffold_template("file-agent", tmp_path)


def test_scaffold_template_allows_explicit_overwrite(tmp_path: Path) -> None:
    (tmp_path / "agent.yaml").write_text("name: existing\n", encoding="utf-8")

    written = scaffold_template("file-agent", tmp_path, overwrite=True)

    assert tmp_path / "agent.yaml" in written
    assert "file-agent" in (tmp_path / "agent.yaml").read_text(encoding="utf-8")


def test_scaffold_template_rejects_unknown_template(tmp_path: Path) -> None:
    with pytest.raises(IrisTemplateNotFoundError):
        scaffold_template("missing-template", tmp_path)


def test_scaffold_file_agent_config_loads_and_builds_tools(tmp_path: Path) -> None:
    scaffold_template("file-agent", tmp_path)

    config = load_agent_config(tmp_path / "agent.yaml")
    registry = build_tool_registry(config.tools)

    assert config.name == "file-agent"
    assert registry.get("read_file").definition.name == "read_file"
    assert registry.get("list_files").definition.name == "list_files"
    assert registry.get("grep_search").definition.name == "grep_search"
