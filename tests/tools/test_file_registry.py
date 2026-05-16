from __future__ import annotations

from iris.tools import register_file_tools
from iris.tools.builtin.file import (
    FILE_TOOL_CLASSES,
    EditFileTool,
    FileTool,
    GrepSearchTool,
    ListFilesTool,
    ReadFileTool,
    WorkspaceFileService,
    WriteFileTool,
)


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


def test_file_tool_classes_define_stable_registration_order() -> None:
    assert FILE_TOOL_CLASSES == (
        ReadFileTool,
        ListFilesTool,
        GrepSearchTool,
        WriteFileTool,
        EditFileTool,
    )


def test_register_file_tools_injects_one_shared_file_service() -> None:
    service = WorkspaceFileService()

    registry = register_file_tools(file_service=service)
    tools = registry.view(include_groups={"file"}).active_tools

    assert len(tools) == 5
    assert all(isinstance(tool, FileTool) for tool in tools)
    assert {id(tool.file_service) for tool in tools} == {id(service)}


def test_concrete_file_tools_use_impl_not_custom_arun() -> None:
    for tool_cls in FILE_TOOL_CLASSES:
        assert "arun" not in tool_cls.__dict__
        assert "_impl" in tool_cls.__dict__
