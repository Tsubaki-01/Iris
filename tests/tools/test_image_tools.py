"""工具图片导入与显式图片回读沿同一文件入口。"""

from io import BytesIO
from pathlib import Path
from threading import get_ident

import pytest
from PIL import Image

from iris.agents import ToolsConfig, build_tool_registry
from iris.exceptions import IrisImageError, IrisToolValidationError
from iris.message import ImageBlock, TextBlock, ToolUseBlock
from iris.tools import (
    DefaultPermissionPolicy,
    ReadFileState,
    ToolCapability,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    WorkspaceFileService,
    import_tool_image,
    register_file_tools,
)
from iris.tools._paths import safe_path_segment


def _image_bytes(size: tuple[int, int] = (32, 16), *, image_format: str = "PNG") -> bytes:
    """构造可检查模型版尺寸的静态图片。"""
    output = BytesIO()
    with Image.new("RGB", size, "blue") as image:
        image.save(output, format=image_format)
    return output.getvalue()


@pytest.mark.parametrize("from_file", [False, True])
def test_tool_image_import_uses_session_cache_and_stable_snapshot(
    tmp_path: Path, from_file: bool
) -> None:
    """普通工具的 bytes 或 workspace 路径共用导入规则，不依赖 provider。"""
    source = tmp_path / "source.png"
    data = _image_bytes()
    source.write_bytes(data)
    context = ToolExecutionContext(workspace_root=tmp_path, session_id="图/工具")
    image = import_tool_image(Path("source.png") if from_file else data, context, name="source")
    source.write_bytes(b"changed")
    assert image.name == "source"
    assert image.original.path.parent == (
        tmp_path / ".iris/image-cache" / safe_path_segment(context.session_id)
    )
    assert image.original == image.model
    assert image.original.path.read_bytes() == data


@pytest.mark.asyncio
@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP"])
async def test_read_file_image_uses_shared_file_service_and_worker_thread(
    tmp_path: Path, image_format: str
) -> None:
    """一次文件服务解析加一次后台读图，结果保留引用和图片块。"""
    from iris.tools.builtin.file import ReadFileTool

    source = tmp_path / "diagram.bin"
    data = _image_bytes(image_format=image_format)
    source.write_bytes(data)
    event_loop_thread = get_ident()
    calls: list[tuple[str, bool]] = []

    class FileService(WorkspaceFileService):
        """记录实际读图使用的已注入服务。"""

        def resolve_path(
            self, path: str, context: ToolExecutionContext, *, write: bool = False
        ) -> Path:
            """读取准备全过程离开主线程，并保留同一文件策略。"""
            assert get_ident() != event_loop_thread
            calls.append((path, write))
            return super().resolve_path(path, context, write=write)

    tool = ReadFileTool(file_service=FileService())
    registry = ToolRegistry()
    registry.register(tool)
    context = ToolExecutionContext(
        workspace_root=tmp_path, session_id="read", read_state=ReadFileState()
    )
    result = await ToolExecutor(
        registry, permission_policy=DefaultPermissionPolicy(write_mode="deny", execute_mode="deny")
    ).execute_one(
        ToolUseBlock(
            id="read",
            name="read_file",
            input={
                "file_path": "diagram.bin",
                "offset": 7,
                "column": 19,
                "limit": 0,
                "with_line_numbers": True,
            },
        ),
        context,
    )
    assert calls == [("diagram.bin", False)]
    assert not result.is_error and result.tool_use_id == "read"
    assert tool.definition.capabilities == {ToolCapability.READ}
    assert tool.is_read_only({}) and tool.is_concurrency_safe({})
    assert tool.definition.context_retention == "observation"
    text, image = result.content
    assert isinstance(text, TextBlock) and isinstance(image, ImageBlock)
    assert str(image.original.path) in text.text and str(image.model.path) in text.text
    assert image.model.mime_type in text.text and "diagram.bin" in text.text
    assert image.model.path != source and image.model.path.read_bytes() == data
    assert context.read_state.files == {}
    assert result.to_msg().tool_results[0].content == result.content


@pytest.mark.asyncio
async def test_read_file_image_reuses_compliant_cache_and_processes_cached_original(
    tmp_path: Path,
) -> None:
    """回读别的 session 引用也不复制合规缓存；超限原图仍按策略生成模型版。"""
    context = ToolExecutionContext(workspace_root=tmp_path, session_id="current")
    original_session = ToolExecutionContext(workspace_root=tmp_path, session_id="original")
    imported = import_tool_image(_image_bytes((2400, 1200)), original_session, name="large")
    executor = ToolExecutor(build_tool_registry(ToolsConfig(builtin=["file.read"])))
    reused = await executor.execute_one(
        ToolUseBlock(id="model", name="read_file", input={"file_path": str(imported.model.path)}),
        context,
    )
    assert not reused.is_error
    model = reused.content[1]
    assert isinstance(model, ImageBlock)
    assert model.original.path == model.model.path == imported.model.path
    assert (model.model.width, model.model.height) == (2000, 1000)
    assert not (tmp_path / ".iris/image-cache" / safe_path_segment("current")).exists()

    processed = await executor.execute_one(
        ToolUseBlock(
            id="original", name="read_file", input={"file_path": str(imported.original.path)}
        ),
        context,
    )
    assert not processed.is_error
    transformed = processed.content[1]
    assert isinstance(transformed, ImageBlock)
    assert transformed.original.path == imported.original.path
    assert transformed.model.path.parent.name == safe_path_segment("current")
    assert transformed.model.path != imported.model.path
    assert (transformed.model.width, transformed.model.height) == (2000, 1000)
    assert transformed.original.path.read_bytes() == _image_bytes((2400, 1200))


def test_cached_path_does_not_skip_image_decoding(tmp_path: Path) -> None:
    """缓存归属只决定是否复用文件，不能让无效文件冒充已准备图片。"""
    source = tmp_path / ".iris/image-cache/previous/invalid.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"\x89PNG\r\n\x1a\ninvalid image")
    with pytest.raises(IrisImageError):
        import_tool_image(source, ToolExecutionContext(workspace_root=tmp_path, session_id="read"))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["missing", "invalid", "file-service-rejection"])
async def test_read_file_image_failures_follow_existing_tool_error_path(
    tmp_path: Path, kind: str
) -> None:
    """文件服务拒绝、源不可读与解码失败都不提交图片成功块。"""
    from iris.tools.builtin.file import ReadFileTool

    source = tmp_path / "image.png"
    if kind == "invalid":
        source.write_bytes(b"\x89PNG\r\n\x1a\nnot image data")

    class FileService(WorkspaceFileService):
        """以相同入口报告已有文件服务的拒绝。"""

        def resolve_path(
            self, path: str, context: ToolExecutionContext, *, write: bool = False
        ) -> Path:
            if kind == "file-service-rejection":
                raise IrisToolValidationError("file service refused")
            return super().resolve_path(path, context, write=write)

    registry = ToolRegistry()
    registry.register(ReadFileTool(file_service=FileService()))
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="failed", name="read_file", input={"file_path": "image.png"}),
        ToolExecutionContext(workspace_root=tmp_path, session_id="read"),
    )
    assert result.is_error
    assert (
        result.error.code
        == {
            "missing": "FILE_NOT_FOUND",
            "invalid": "EXECUTION_ERROR",
            "file-service-rejection": "VALIDATION_ERROR",
        }[kind]
    )
    assert not any(isinstance(block, ImageBlock) for block in result.content)


def test_default_file_registration_keeps_existing_names() -> None:
    """图片沿已有读取入口，不增加新工具或别名。"""
    assert {tool.name for tool in register_file_tools().view().active_tools} == {
        "read_file",
        "list_files",
        "grep_search",
        "write_file",
        "edit_file",
    }


def test_sync_file_service_returns_image_without_text_read_record(tmp_path: Path) -> None:
    """直接调用同步文件服务也交付图片，不能尝试合并空的文本读取观测。"""
    source = tmp_path / "image.bin"
    source.write_bytes(_image_bytes())
    context = ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState())
    service = WorkspaceFileService()
    from iris.tools import ReadFileInput

    image = service.read_file(ReadFileInput(file_path="image.bin"), context, max_chars=2000)
    assert isinstance(image, ImageBlock)
    assert image.model.path.read_bytes() == _image_bytes()
    assert context.read_state.files == {}


@pytest.mark.asyncio
async def test_image_suffix_does_not_change_text_paging_or_read_state(tmp_path: Path) -> None:
    """内容识别保留普通UTF-8文本的分页、行号和文本编辑读取状态。"""
    source = tmp_path / "notes.png"
    source.write_text("first\nsecond\nthird\n", encoding="utf-8")
    context = ToolExecutionContext(
        workspace_root=tmp_path, session_id="text", read_state=ReadFileState()
    )
    result = await ToolExecutor(register_file_tools()).execute_one(
        ToolUseBlock(
            id="text",
            name="read_file",
            input={
                "file_path": "notes.png",
                "offset": 1,
                "column": 2,
                "limit": 1,
                "with_line_numbers": True,
            },
        ),
        context,
    )
    assert not result.is_error
    assert result.model_content.startswith("L0002 | cond\n")
    assert "next_offset=2, next_column=0; has_more=true" in result.model_content
    assert all(isinstance(block, TextBlock) for block in result.content)
    assert str(source) in context.read_state.files


@pytest.mark.asyncio
async def test_read_file_rejects_invalid_cached_image_through_tool_error(tmp_path: Path) -> None:
    """有缓存路径和图片头也必须经过真实解码，不把损坏图片交给provider。"""
    source = tmp_path / ".iris/image-cache/old/broken.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"\x89PNG\r\n\x1a\nbroken cache")
    result = await ToolExecutor(register_file_tools()).execute_one(
        ToolUseBlock(id="broken", name="read_file", input={"file_path": str(source)}),
        ToolExecutionContext(workspace_root=tmp_path, session_id="current"),
    )
    assert result.is_error and result.error.code == "EXECUTION_ERROR"
    assert not any(isinstance(block, ImageBlock) for block in result.content)
