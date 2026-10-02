"""图片文件引用和有序数据块的消息契约。"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from iris.message import (
    ImageBlock,
    ImageFileRef,
    Msg,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    image_block_from_saved,
    image_reference_text,
)


def _image() -> ImageBlock:
    return ImageBlock(
        original=ImageFileRef(
            path=Path.cwd() / "original.png", mime_type="image/png", width=3000, height=1500
        ),
        model=ImageFileRef(
            path=Path.cwd() / "model.jpg", mime_type="image/jpeg", width=2000, height=1000
        ),
        name="截图",
    )


def test_images_roundtrip_at_top_level_and_inside_tool_result() -> None:
    image = _image()
    message = Msg(
        role=Role.USER,
        content=[
            TextBlock(text="看图"),
            image,
            ToolResultBlock(
                tool_use_id="call-1",
                content=[TextBlock(text="之前"), image, TextBlock(text="之后")],
            ),
        ],
    )
    serialized = message.model_dump_json()
    assert "base64" not in serialized
    for restored in (Msg.model_validate_json(serialized), Msg.from_dict(json.loads(serialized))):
        assert restored == message
        assert isinstance(restored.blocks[1], ImageBlock)
        assert restored.blocks[1].model.path == image.model.path
        assert restored.text == "看图"
        result = restored.tool_results[0]
        assert result.text == "之前\n之后"
        assert [block.type for block in result.content] == ["text", "image", "text"]


def test_tool_result_factory_wraps_strings_and_preserves_data_blocks() -> None:
    assert Msg.tool_result(tool_use_id="c", content="done").tool_results[0].content == [
        TextBlock(text="done")
    ]
    blocks = [TextBlock(text="结果"), _image()]
    assert Msg.tool_result(tool_use_id="c", content=blocks).tool_results[0].content == blocks
    with pytest.raises(ValidationError):
        ToolResultBlock(tool_use_id="c", content="old string")
    with pytest.raises(ValidationError):
        ToolResultBlock(tool_use_id="c", content=[ToolUseBlock(name="nested")])


def test_image_reference_locates_both_versions_without_reading_pixels() -> None:
    image = _image()
    text = image_reference_text(image)
    for value in (
        "[image:",
        "截图",
        str(image.original.path),
        str(image.model.path),
        "image/png",
        "image/jpeg",
        "3000x1500",
        "2000x1000",
    ):
        assert value in text
    assert "original.png" in image_reference_text(image.model_copy(update={"name": None}))


@pytest.mark.parametrize("transformed", [False, True])
def test_saved_image_projection_uses_returned_file_information(transformed: bool) -> None:
    from iris.utils.images import SavedImage, SavedImageFile

    original = SavedImageFile(Path.cwd() / "original.png", "image/png", 3000, 1500)
    model = (
        SavedImageFile(Path.cwd() / "model.jpg", "image/jpeg", 2000, 1000)
        if transformed
        else original
    )
    block = image_block_from_saved(SavedImage(original, model), name="diagram")
    assert block.name == "diagram"
    assert block.original.path == original.path
    assert block.model.model_dump() == {
        "path": model.path,
        "mime_type": model.mime_type,
        "width": model.width,
        "height": model.height,
    }
    if not transformed:
        assert block.model is block.original
