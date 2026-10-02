"""Hook 反馈沿唯一工具正文投影和既有 artifact 预算交付。"""

from pathlib import Path

import pytest

from iris.message import (
    ImageBlock,
    ImageFileRef,
    Role,
    TextBlock,
    ToolResultBlock,
    image_reference_text,
)
from iris.tools import ToolArtifact, ToolErrorInfo, ToolResult
from iris.tools.artifacts import ToolArtifactStore, truncate_tool_result


@pytest.mark.parametrize("failed", [False, True])
def test_feedback_uses_one_result_block_and_preserves_body_fields(failed: bool) -> None:
    result = ToolResult(
        tool_use_id="call",
        tool_name="read",
        content=[TextBlock(text="first line"), TextBlock(text="second line")],
        is_error=failed,
        error=ToolErrorInfo(code="READ_FAILED", message="cannot read", details={"line": 7})
        if failed
        else None,
        data={"value": 7},
        hook_feedback=("first feedback", "second feedback"),
    )

    body = "Error[READ_FAILED]: cannot read" if failed else "first line\nsecond line"
    expected = body + "\n[Hook feedback]\nfirst feedback\n[Hook feedback]\nsecond feedback"
    message = result.to_msg()

    assert result.model_content == expected
    assert message.role is Role.USER
    assert len(message.content) == 1
    assert isinstance(message.content[0], ToolResultBlock)
    assert message.content[0].text == expected
    assert result.content == [TextBlock(text="first line"), TextBlock(text="second line")]
    assert result.data == {"value": 7}
    if result.error is not None:
        assert result.error.message == "cannot read"
        assert result.error.details == {"line": 7}


@pytest.mark.parametrize("failed", [False, True])
def test_small_feedback_retains_structured_result_without_artifact(
    tmp_path: Path, failed: bool
) -> None:
    result = ToolResult(
        tool_use_id="small",
        tool_name="read",
        content=[TextBlock(text="body")],
        is_error=failed,
        error=ToolErrorInfo(code="FAILED", message="error body") if failed else None,
        hook_feedback=("note",),
    )
    store = ToolArtifactStore(tmp_path / "artifacts")

    limited = store.persist_if_large(result, max_chars=1000)

    assert limited is result
    assert limited.hook_feedback == ("note",)
    assert limited.artifact is None
    assert not store.root.exists()


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("with_image", [False, True])
def test_large_feedback_is_persisted_once_and_preview_is_bounded(
    tmp_path: Path, failed: bool, native: bool, with_image: bool
) -> None:
    native_path = tmp_path / "image.png"
    if native:
        native_path.write_bytes(b"native-image")
    original = "body line\n" * 1000
    feedback = ("FIRST_FEEDBACK", "LAST_FEEDBACK")
    ref = ImageFileRef(path=tmp_path / "image.png", mime_type="image/png", width=1, height=1)
    image = ImageBlock(original=ref, model=ref)
    result = ToolResult(
        tool_use_id="large",
        tool_name="read",
        content=[image, TextBlock(text=original)] if with_image else [TextBlock(text=original)],
        is_error=failed,
        error=ToolErrorInfo(code="READ_FAILED", message=original, details={"line": 7})
        if failed
        else None,
        data={"value": 7},
        artifact=ToolArtifact(path=native_path, mime_type="image/png", size_bytes=12)
        if native
        else None,
        hook_feedback=feedback,
    )
    full = ("Error[READ_FAILED]: " if failed else "") + original
    full += "\n[Hook feedback]\nFIRST_FEEDBACK\n[Hook feedback]\nLAST_FEEDBACK"
    if with_image:
        full = f"{image_reference_text(image)}\n\n{full}"
    store = ToolArtifactStore(tmp_path / "artifacts", preview_chars=800, preview_mode="head_tail")

    limited = store.persist_if_large(result, max_chars=1600)

    assert limited.artifact is not None and limited.artifact.text_path is not None
    assert limited.artifact.text_path.read_text(encoding="utf-8") == full
    assert len(list(store.root.iterdir())) == 1
    assert len(limited.model_content) <= 1600
    assert limited.hook_feedback == ()
    assert limited.model_content.count("FIRST_FEEDBACK") == 1
    assert limited.model_content.count("LAST_FEEDBACK") == 1
    assert limited.to_msg().tool_results[0].text == limited.model_content
    assert [block for block in limited.model_blocks if isinstance(block, ImageBlock)] == (
        [image] if with_image else []
    )
    assert str(limited.artifact.text_path) in limited.model_content
    assert limited.data == result.data
    assert result.hook_feedback == feedback
    if failed:
        assert limited.model_content.startswith("Error[READ_FAILED]: ")
        assert limited.model_content.count("Error[READ_FAILED]: ") == 1
        assert limited.error is not None and limited.error.details == {"line": 7}
    if native:
        assert limited.artifact.path == native_path
        assert native_path.read_bytes() == b"native-image"
        assert limited.artifact.text_path != native_path
    else:
        assert limited.artifact.path == limited.artifact.text_path


@pytest.mark.parametrize("failed", [False, True])
def test_feedback_alone_counts_toward_output_budget(tmp_path: Path, failed: bool) -> None:
    result = ToolResult(
        tool_use_id="feedback-large",
        tool_name="read",
        content=[TextBlock(text="short body")],
        is_error=failed,
        error=ToolErrorInfo(code="FAILED", message="short error") if failed else None,
        hook_feedback=("feedback " * 1000,),
    )
    store = ToolArtifactStore(tmp_path / "artifacts", preview_chars=20)

    limited = store.persist_if_large(result, max_chars=1200)

    assert limited.artifact is not None
    assert limited.artifact.path.read_text(encoding="utf-8") == result.model_content
    assert limited.hook_feedback == ()
    assert len(limited.model_content) <= 1200


@pytest.mark.parametrize("failed", [False, True])
def test_direct_truncation_includes_feedback_with_one_error_prefix(failed: bool) -> None:
    result = ToolResult(
        tool_use_id="call",
        tool_name="read",
        content=[TextBlock(text="body")],
        is_error=failed,
        error=ToolErrorInfo(code="FAILED", message="body") if failed else None,
        hook_feedback=("first feedback", "second feedback"),
    )

    limited = truncate_tool_result(result, max_chars=200, preview_chars=200, suffix="\n[more]")

    assert limited.model_content == result.model_content + "\n[more]"
    assert limited.hook_feedback == ()
    assert len(limited.model_content) <= 200
