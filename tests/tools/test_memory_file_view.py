"""记忆文件读取范围与陈旧状态必须在真实文件工具路径中生效。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from iris.agents import ToolsConfig, build_tool_registry
from iris.exceptions import IrisToolValidationError
from iris.memory import MemoryConfig, MemoryWriteInput, build_memory_service_from_config
from iris.message import ToolUseBlock
from iris.tools import (
    GrepSearchInput,
    ListFilesInput,
    ReadFileInput,
    ToolExecutionContext,
    ToolExecutor,
)
from iris.tools.builtin import file as file_module
from iris.tools.builtin.file import WorkspaceFileService, WriteFileInput


@dataclass
class _View:
    """提供可在读取途中推进版本的投影视图。"""

    root: Path
    namespaces: tuple[str, ...] = ("project",)
    current_revision: int = 2

    @property
    def document_paths(self) -> tuple[Path, ...]:
        return tuple(self.namespace_directory(ns) / "notes.md" for ns in self.namespaces)

    def namespace_directory(self, namespace: str) -> Path:
        return self.root / "namespaces" / namespace

    def source_revision(self, first_line: str) -> int | None:
        return int(first_line.removeprefix("revision: ").strip())

    def projection_warning(self, namespace: str) -> str | None:
        """此替身只有文件来源落后，没有未完成的整组正文发布。"""
        return None

    def warning(
        self, namespace: str, source_revision: int | None, *, overview: bool = False
    ) -> str | None:
        return "MEMORY_STALE: 文件未同步" if source_revision != self.current_revision else None


def _fixture(tmp_path: Path) -> tuple[WorkspaceFileService, ToolExecutionContext, _View, Path]:
    view = _View(tmp_path / ".iris" / "memory")
    allowed = view.document_paths[0]
    allowed.parent.mkdir(parents=True)
    allowed.write_text("revision: 1\nold value\n", encoding="utf-8")
    excluded = view.root / "namespaces" / "private" / "notes.md"
    excluded.parent.mkdir(parents=True)
    excluded.write_text("revision: 2\nsecret value\n", encoding="utf-8")
    (view.root / "legacy.md").write_text("mixed namespace value", encoding="utf-8")
    context = ToolExecutionContext(workspace_root=tmp_path)
    return WorkspaceFileService(memory_view=view), context, view, allowed


def test_direct_read_and_recursive_search_apply_memory_scope(tmp_path: Path) -> None:
    service, context, view, allowed = _fixture(tmp_path)
    for excluded in (view.root / "legacy.md", view.root / "namespaces/private/notes.md"):
        with pytest.raises(IrisToolValidationError, match="记忆投影范围"):
            service.read_file(ReadFileInput(file_path=str(excluded)), context, max_chars=2000)
    listing = service.list_files(ListFilesInput(path=str(view.root)), context)
    assert "project" in listing
    assert "private" not in listing and "legacy" not in listing
    result = service.grep_search(GrepSearchInput(path=str(view.root), pattern="value"), context)
    assert "old value" in result and "MEMORY_STALE" in result
    assert "secret" not in result and "mixed" not in result


def test_read_and_empty_grep_report_stale_files(tmp_path: Path) -> None:
    service, context, view, allowed = _fixture(tmp_path)
    result = service.read_file(ReadFileInput(file_path=str(allowed)), context, max_chars=2000)
    assert "old value" in result and "MEMORY_STALE" in result
    result = service.grep_search(GrepSearchInput(path=str(view.root), pattern="new value"), context)
    assert "MEMORY_STALE" in result
    view.current_revision = 1
    result = service.grep_search(GrepSearchInput(path=str(view.root), pattern="new value"), context)
    assert result == ""


def test_projection_writes_use_memory_service_and_other_files_stay_normal(tmp_path: Path) -> None:
    service, context, view, allowed = _fixture(tmp_path)
    with pytest.raises(IrisToolValidationError, match="memory 工具"):
        service.write_file(WriteFileInput(file_path=str(allowed), content="replacement"), context)
    service.write_file(WriteFileInput(file_path="normal.md", content="normal"), context)
    result = service.read_file(ReadFileInput(file_path="normal.md"), context, max_chars=2000)
    assert result.startswith("normal\n\n[read_file:")
    assert "MEMORY_STALE" not in result


def test_read_checks_memory_revision_after_consuming_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """读取过程中记忆版本推进时，返回的旧内容仍带陈旧提示。"""
    service, context, view, allowed = _fixture(tmp_path)
    view.current_revision = 1
    read_page = file_module.read_text_page

    def read_and_advance(*args: Any, **kwargs: Any) -> str:
        text = read_page(*args, **kwargs)
        view.current_revision = 2
        return text

    monkeypatch.setattr(file_module, "read_text_page", read_and_advance)
    result = service.read_file(ReadFileInput(file_path=str(allowed)), context, max_chars=2000)
    assert "old value" in result and "MEMORY_STALE" in result


@pytest.mark.asyncio
async def test_configured_file_tools_read_real_projection_and_report_failed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实memory服务的目录、版本与投影失败沿文件工具返回给模型。"""
    config = MemoryConfig(backend="sqlite", read_namespaces=["project"])
    service = build_memory_service_from_config(config, tmp_path)
    assert service is not None and service.mirror is not None
    service.remember(
        MemoryWriteInput(
            namespace="project",
            category="reference",
            kind="fact",
            text="endpoint A",
            reason="fixture",
        )
    )
    service.remember(
        MemoryWriteInput(
            namespace="private",
            category="reference",
            kind="fact",
            text="hidden value",
            reason="fixture",
        )
    )
    view = service.file_access(config.read_namespaces)
    assert view is not None
    target = view.namespace_directory("project") / "Reference/notes.md"
    registry = build_tool_registry(
        ToolsConfig(builtin=["file.read", "file.grep"]),
        memory_service=service,
        memory_config=config,
    )
    assert {tool.name for tool in registry.view().active_tools} == {
        "read_file",
        "grep_search",
        "memory_search",
        "memory_list",
        "memory_get",
    }
    executor = ToolExecutor(registry)
    context = ToolExecutionContext(workspace_root=tmp_path)
    result = await executor.execute_one(
        ToolUseBlock(id="read-current", name="read_file", input={"file_path": str(target)}), context
    )
    assert not result.is_error and "endpoint A" in result.model_content
    assert "陈旧" not in result.model_content

    def fail_projection(*args: Any, **kwargs: Any) -> None:
        raise OSError("projection unavailable")

    monkeypatch.setattr(service.mirror, "rebuild_from_store", fail_projection)
    service.remember(
        MemoryWriteInput(
            namespace="project",
            category="reference",
            kind="fact",
            text="endpoint B",
            reason="fixture",
        )
    )
    result = await executor.execute_one(
        ToolUseBlock(
            id="search-stale",
            name="grep_search",
            input={"path": str(view.root), "pattern": "endpoint B"},
        ),
        context,
    )
    assert not result.is_error and "未同步" in result.model_content
    assert "hidden value" not in result.model_content


@pytest.mark.asyncio
async def test_first_failed_projection_does_not_look_like_an_empty_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """首次发布未产生任何文件时，搜索仍明确数据库已保存但正文未同步。"""
    config = MemoryConfig(backend="sqlite")
    service = build_memory_service_from_config(config, tmp_path)
    assert service is not None and service.mirror is not None

    def fail_projection(*args: Any, **kwargs: Any) -> None:
        raise OSError("first projection unavailable")

    monkeypatch.setattr(service.mirror, "rebuild_from_store", fail_projection)
    service.remember(
        MemoryWriteInput(text="已保存的 endpoint", category="reference", reason="fixture")
    )
    registry = build_tool_registry(
        ToolsConfig(builtin=["file.read", "file.grep"]),
        memory_service=service,
        memory_config=config,
    )
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(
            id="search-unpublished",
            name="grep_search",
            input={"path": str(service.mirror.root), "pattern": "endpoint"},
        ),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert not result.is_error and "正文文件未同步" in result.model_content
    missing = service.mirror.namespace_directory("project") / "Reference/notes.md"
    for name, arguments in (
        ("read_file", {"file_path": str(missing)}),
        ("grep_search", {"path": str(missing.parent), "pattern": "endpoint"}),
    ):
        result = await ToolExecutor(registry).execute_one(
            ToolUseBlock(id=f"missing-{name}", name=name, input=arguments),
            ToolExecutionContext(workspace_root=tmp_path),
        )
        assert result.is_error and "正文文件未同步" in result.model_content
