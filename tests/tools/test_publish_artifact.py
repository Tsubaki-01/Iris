"""显式文件发布保存独立副本，并复用文件边界和现有工具结果协议。"""

import base64
from pathlib import Path
from typing import BinaryIO

import pytest

from iris.agents import ToolsConfig, build_tool_registry
from iris.exceptions import IrisToolExecutionError
from iris.message import ToolUseBlock
from iris.tools import (
    DefaultPermissionPolicy,
    ReadFileState,
    ToolCapability,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolTimeoutOwner,
    register_file_tools,
)
from iris.tools.artifacts import ToolArtifactStore

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)


def test_store_copies_unique_snapshots_and_keeps_extension(tmp_path: Path) -> None:
    """相同调用 id 的再次发布不会覆盖旧副本，源文件删除也不影响交付。"""
    source = tmp_path / "report.csv"
    source.write_bytes(b"name,value\na,1\n")
    store = ToolArtifactStore(tmp_path / "results")
    first = store.persist_file("same", source, preview="report")
    source.write_bytes(b"name,value\nb,2\n")
    second = store.persist_file("same", source, preview="report again")
    source.unlink()
    assert first.path != second.path
    assert first.path.suffix == second.path.suffix == ".csv"
    assert first.path.read_bytes() == b"name,value\na,1\n"
    assert second.path.read_bytes() == b"name,value\nb,2\n"
    assert first.size_bytes == len(first.path.read_bytes())
    assert first.text_path is None and second.text_path is None
    assert first.preview == "report"


def test_partial_copy_failure_removes_only_current_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """复制中途失败返回 artifact 错误，保留其他已发布文件。"""
    source = tmp_path / "report.csv"
    source.write_bytes(b"some data")
    store = ToolArtifactStore(tmp_path / "results")
    existing = store.persist_file("same", source, preview="first")
    calls = []

    def fail_copy(origin: BinaryIO, target: BinaryIO, length: int) -> None:
        calls.append(length)
        target.write(origin.read(3))
        raise OSError("disk full")

    monkeypatch.setattr("iris.tools.artifacts.shutil.copyfileobj", fail_copy)
    with pytest.raises(IrisToolExecutionError, match="ARTIFACT_ERROR"):
        store.persist_file("same", source, preview="second")
    assert calls and len(calls) == 1 and calls[0] > 0
    assert list(store.root.iterdir()) == [existing.path]
    assert existing.path.read_bytes() == b"some data"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,payload,mimes",
    [
        ("plot.png", _PNG, {"image/png"}),
        (
            "report.csv",
            b"name,value\na,1\n",
            {"text/csv", "application/csv", "application/vnd.ms-excel"},
        ),
        ("binary.irisunknown", b"\x00\xff\x01", {"application/octet-stream"}),
    ],
)
async def test_publish_accepts_binary_without_read_state_and_uses_read_permission(
    tmp_path: Path, name: str, payload: bytes, mimes: set[str]
) -> None:
    """READ 发布不依赖执行/写权限，也不把发布当作编辑前的文件阅读。"""
    source = tmp_path / name
    source.write_bytes(payload)
    registry = build_tool_registry(ToolsConfig(builtin=["file.publish"]))
    tool = registry.get("publish_artifact")
    assert tool.definition.capabilities == {ToolCapability.READ}
    assert tool.definition.group == "file" and tool.definition.context_retention == "keep"
    assert tool.timeout_owner is ToolTimeoutOwner.RUNTIME
    context = ToolExecutionContext(
        workspace_root=tmp_path, session_id="session", read_state=ReadFileState()
    )
    result = await ToolExecutor(
        registry, permission_policy=DefaultPermissionPolicy(write_mode="deny", execute_mode="deny")
    ).execute_one(
        ToolUseBlock(id="publish", name="publish_artifact", input={"file_path": name}), context
    )
    assert not result.is_error and result.artifact is not None
    artifact = result.artifact
    assert artifact.path.is_absolute() and artifact.path != source
    artifact.path.relative_to(tmp_path / ".iris" / "tool-results")
    assert artifact.path.read_bytes() == payload
    assert artifact.size_bytes == len(payload)
    assert artifact.mime_type in mimes
    assert artifact.text_path is None
    assert name in result.model_content and artifact.mime_type in result.model_content
    assert context.read_state.files == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_path,error_code",
    [("missing", "FILE_NOT_FOUND"), (".", "FILE_NOT_FOUND"), ("../outside", "VALIDATION_ERROR")],
)
async def test_unavailable_publish_never_returns_artifact(
    tmp_path: Path, file_path: str, error_code: str
) -> None:
    """缺文件、目录和既有路径边界拒绝都不伪造成功产物。"""
    executor = ToolExecutor(build_tool_registry(ToolsConfig(builtin=["file.publish"])))
    result = await executor.execute_one(
        ToolUseBlock(id="missing", name="publish_artifact", input={"file_path": file_path}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert result.error is not None and result.error.code == error_code
    assert result.artifact is None
    assert not (tmp_path / ".iris").exists()


def test_publish_requires_explicit_sdk_registration_and_keeps_file_input_policy() -> None:
    """默认文件工具集合不扩张，SDK 可按相同文件参数策略显式注册。"""
    from iris.tools import PublishArtifactTool

    assert "publish_artifact" not in {
        tool.name for tool in register_file_tools().view().active_tools
    }
    registry = ToolRegistry()
    tool = PublishArtifactTool()
    registry.register(tool)
    assert registry.get("publish_artifact") is tool
    assert tool.validate_input({"file_path": "report.csv", "extra": "ignored"}).model_dump() == {
        "file_path": "report.csv"
    }
