from __future__ import annotations

from iris.tools import register_file_tools


def test_file_tool_registry_view_exposes_file_group_only() -> None:
    registry = register_file_tools()

    schemas = registry.view(include_groups={"file"}).active_schemas()

    assert [schema["name"] for schema in schemas] == [
        "read_file",
        "list_files",
        "grep_search",
        "write_file",
        "edit_file",
    ]
