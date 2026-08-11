from __future__ import annotations

from pathlib import Path

import pytest

from iris.agents import PythonToolsConfig, ToolsConfig, build_tool_registry


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
