from __future__ import annotations

from pathlib import Path

import pytest

from iris.agents import PythonToolsConfig, ToolsConfig, build_tool_registry, load_agent_config


def test_build_tool_registry_registers_python_function_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "user_tools.py"
    module_path.write_text(
        '''
def search_notes(query: str) -> str:
    """搜索本地笔记。"""
    return f"search: {query}"
''',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = build_tool_registry(
        ToolsConfig(python=PythonToolsConfig(functions=["user_tools:search_notes"]))
    )

    assert registry.get("search_notes").definition.description == "搜索本地笔记。"


def test_subagent_path_is_parent_relative_and_ordinary_registry_ignores_it(tmp_path: Path) -> None:
    assert ToolsConfig().subagent is None
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "name: parent\nmodel: openai/test\nsystem: parent\n"
        "tools:\n  builtin: [file.read]\n  subagent: catalogs/agents.yaml\n",
        encoding="utf-8",
    )
    config = load_agent_config(config_path)
    assert config.tools.subagent == tmp_path / "catalogs" / "agents.yaml"
    registry = build_tool_registry(config.tools)
    assert registry.get("read_file").name == "read_file"
