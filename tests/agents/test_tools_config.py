from __future__ import annotations

from pathlib import Path

import pytest

from iris.agents import PythonToolsConfig, ToolsConfig, build_tool_registry
from iris.exceptions import IrisConfigError, IrisToolNotFoundError


def test_build_tool_registry_registers_named_builtin_tools() -> None:
    registry = build_tool_registry(
        ToolsConfig(
            builtin=["file.read", "file.list", "file.grep", "file.write", "file.edit"]
        )
    )

    assert registry.get("read_file").definition.name == "read_file"
    assert registry.get("list_files").definition.name == "list_files"
    assert registry.get("grep_search").definition.name == "grep_search"
    assert registry.get("write_file").definition.name == "write_file"
    assert registry.get("edit_file").definition.name == "edit_file"


def test_build_tool_registry_only_registers_requested_builtin_tools() -> None:
    registry = build_tool_registry(ToolsConfig(builtin=["file.read"]))

    assert registry.get("read_file").definition.name == "read_file"
    with pytest.raises(IrisToolNotFoundError):
        registry.get("write_file")


def test_build_tool_registry_rejects_unknown_builtin_tool() -> None:
    with pytest.raises(IrisConfigError, match="file.unknown"):
        build_tool_registry(ToolsConfig(builtin=["file.unknown"]))


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
        ToolsConfig(
            python=PythonToolsConfig(
                functions=["user_tools:search_notes"],
            )
        )
    )

    assert registry.get("search_notes").definition.description == "搜索本地笔记。"


def test_build_tool_registry_registers_python_registrars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "user_registrar.py"
    module_path.write_text(
        '''
def search_notes(query: str) -> str:
    """搜索本地笔记。"""
    return query

def create_note(title: str) -> str:
    """创建本地笔记。"""
    return title

def register_tools(registry):
    registry.register_function(search_notes)
    registry.register_function(create_note)
''',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = build_tool_registry(
        ToolsConfig(
            python=PythonToolsConfig(
                registrars=["user_registrar:register_tools"],
            )
        )
    )

    assert registry.get("search_notes").definition.name == "search_notes"
    assert registry.get("create_note").definition.name == "create_note"


@pytest.mark.parametrize(
    "ref",
    [
        "missing_colon",
        ":function",
        "module:",
        "missing_module:search",
        "math:missing_function",
        "math:pi",
    ],
)
def test_build_tool_registry_rejects_bad_python_function_refs(ref: str) -> None:
    with pytest.raises(IrisConfigError, match=ref.replace("\\", "\\\\")):
        build_tool_registry(ToolsConfig(python=PythonToolsConfig(functions=[ref])))


def test_build_tool_registry_rejects_incompatible_registrar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "bad_registrar.py"
    module_path.write_text(
        """
def register_tools() -> None:
    return None
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(IrisConfigError, match="bad_registrar:register_tools"):
        build_tool_registry(
            ToolsConfig(
                python=PythonToolsConfig(
                    registrars=["bad_registrar:register_tools"],
                )
            )
        )
