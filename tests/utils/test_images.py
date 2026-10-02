"""共享图片准备和本地文件副本的可观察行为。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from random import Random
from typing import Any, BinaryIO

import pytest
from PIL import Image

from iris.exceptions import IrisImageError
from iris.utils import images
from iris.utils.images import prepare_image, save_image


def _image_bytes(
    *,
    image_format: str = "PNG",
    mode: str = "RGB",
    size: tuple[int, int] = (64, 32),
    noise: bool = False,
    orientation: int | None = None,
) -> bytes:
    """生成小型确定性图像，避免依赖外部素材。"""
    if noise:
        image = Image.frombytes(mode, size, Random(7).randbytes(size[0] * size[1] * len(mode)))
    else:
        image = Image.new(mode, size, (30, 120, 210, 90) if mode == "RGBA" else (30, 120, 210))
    output = BytesIO()
    options: dict[str, Any] = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        options["exif"] = exif
    image.save(output, format=image_format, **options)
    image.close()
    return output.getvalue()


@pytest.mark.parametrize(
    "image_format,mime",
    [
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),
        ("WEBP", "image/webp"),
    ],
)
def test_small_images_keep_original_encoding_without_upscale(image_format: str, mime: str) -> None:
    data = _image_bytes(image_format=image_format)
    prepared = prepare_image(data)
    assert prepared.model is prepared.original
    assert prepared.original.data == data
    assert (prepared.model.width, prepared.model.height) == (64, 32)
    assert prepared.model.mime_type == mime


def test_saved_source_reuse_still_prepares_image_once_without_new_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """已保存的合规图片仍经过解码，但不创建重复文件或目标目录。"""
    source = tmp_path / "cached.png"
    source.write_bytes(_image_bytes())
    prepare = images.prepare_image
    prepared_data: list[bytes] = []

    def prepare_once(data: bytes) -> images.PreparedImage:
        prepared_data.append(data)
        return prepare(data)

    monkeypatch.setattr(images, "prepare_image", prepare_once)
    target = tmp_path / "next-session"
    saved = save_image(source, cache_dir=target, reuse_source=True)
    assert saved.original is saved.model
    assert saved.model.path == source
    assert len(prepared_data) == 1 and prepared_data[0] == source.read_bytes()
    assert not target.exists()


def test_saved_source_reuse_keeps_original_and_writes_only_transformed_model(
    tmp_path: Path,
) -> None:
    """原缓存图需缩放时只生成模型版，原始路径和字节保持不变。"""
    source = tmp_path / "cached-large.png"
    data = _image_bytes(size=(2400, 1200))
    source.write_bytes(data)
    target = tmp_path / "next-session"
    saved = save_image(source, cache_dir=target, reuse_source=True)
    assert saved.original.path == source and source.read_bytes() == data
    assert saved.model.path.parent == target
    assert list(target.iterdir()) == [saved.model.path]
    assert (saved.model.width, saved.model.height) == (2000, 1000)


def test_large_png_is_scaled_proportionally_and_keeps_original_bytes() -> None:
    data = _image_bytes(size=(2400, 1200))
    prepared = prepare_image(data)
    assert prepared.original.data == data
    assert (prepared.original.width, prepared.original.height) == (2400, 1200)
    assert (prepared.model.width, prepared.model.height) == (2000, 1000)
    assert prepared.model.mime_type == "image/png"
    with Image.open(BytesIO(prepared.model.data)) as model:
        assert model.size == (2000, 1000)
        assert model.format == "PNG"


def test_exif_orientation_applies_only_to_model_copy() -> None:
    data = _image_bytes(image_format="JPEG", orientation=6)
    prepared = prepare_image(data)
    assert prepared.original.data == data
    assert (prepared.original.width, prepared.original.height) == (64, 32)
    assert (prepared.model.width, prepared.model.height) == (32, 64)
    with Image.open(BytesIO(prepared.model.data)) as model:
        assert model.getexif().get(274, 1) == 1
        assert model.format == "JPEG"


def test_alpha_survives_processing_without_palette_or_background() -> None:
    data = _image_bytes(mode="RGBA", size=(2400, 1200))
    prepared = prepare_image(data)
    assert prepared.original.data == data
    assert prepared.model.mime_type == "image/png"
    with Image.open(BytesIO(prepared.model.data)) as model:
        assert model.mode == "RGBA"
        assert model.getchannel("A").getextrema() == (90, 90)
        assert model.size == (2000, 1000)


def test_opaque_png_can_use_jpeg_quality_to_fit_without_further_downscale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _image_bytes(size=(128, 96), noise=True)
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 4000)
    prepared = prepare_image(data)
    assert prepared.original.data == data
    assert len(prepared.model.data) <= 4000
    assert prepared.model.mime_type == "image/jpeg"
    assert (prepared.model.width, prepared.model.height) == (128, 96)
    with Image.open(BytesIO(prepared.model.data)) as model:
        assert model.format == "JPEG"


def test_alpha_uses_bounded_extra_downscales_to_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _image_bytes(mode="RGBA", size=(128, 96), noise=True)
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 4500)
    prepared = prepare_image(data)
    assert len(prepared.model.data) <= 4500
    assert prepared.model.mime_type == "image/png"
    assert (prepared.model.width, prepared.model.height) == (32, 24)
    with Image.open(BytesIO(prepared.model.data)) as model:
        assert model.mode == "RGBA"


def test_exhausted_candidates_fail_instead_of_returning_oversized_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(IrisImageError, match="仍超过"):
        prepare_image(_image_bytes(mode="RGBA", size=(16, 12)))


@pytest.mark.parametrize("data", [b"not an image", _image_bytes(image_format="BMP")])
def test_invalid_or_unsupported_image_fails(data: bytes) -> None:
    with pytest.raises(IrisImageError):
        prepare_image(data)


def test_animated_image_is_out_of_scope() -> None:
    output = BytesIO()
    with Image.new("RGB", (8, 8), "red") as first, Image.new("RGB", (8, 8), "blue") as second:
        first.save(output, format="PNG", save_all=True, append_images=[second], duration=100)
    with pytest.raises(IrisImageError, match="静态"):
        prepare_image(output.getvalue())


@pytest.mark.parametrize("from_file", [False, True])
def test_save_stable_snapshot_detects_real_format_and_does_not_overwrite(
    tmp_path: Path,
    from_file: bool,
) -> None:
    data = _image_bytes()
    source = tmp_path / "misleading.jpg"
    source.write_bytes(data)
    cache = tmp_path / "chosen-cache"
    first = save_image(source if from_file else data, cache_dir=cache)
    second = save_image(source if from_file else data, cache_dir=cache)
    source.write_bytes(b"source changed after import")
    assert first.model is first.original
    assert first.original.path != second.original.path
    assert first.original.path.is_absolute()
    assert first.original.path.parent == cache.resolve()
    assert first.original.path.suffix == ".png"
    assert first.original.path.read_bytes() == second.original.path.read_bytes() == data
    assert len(list(cache.iterdir())) == 2


def test_transformed_save_preserves_original_and_model_under_one_asset(tmp_path: Path) -> None:
    data = _image_bytes(image_format="JPEG", orientation=6)
    saved = save_image(data, cache_dir=tmp_path / "cache")
    assert saved.original.path != saved.model.path
    assert saved.original.path.name.split(".")[0] == saved.model.path.name.split(".")[0]
    assert saved.original.path.read_bytes() == data
    with Image.open(saved.original.path) as original, Image.open(saved.model.path) as model:
        assert original.size == (64, 32)
        assert model.size == (32, 64)
        assert model.getexif().get(274, 1) == 1


def test_read_or_decode_failure_does_not_publish_files(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    with pytest.raises(IrisImageError):
        save_image(tmp_path / "missing.png", cache_dir=cache)
    with pytest.raises(IrisImageError):
        save_image(b"invalid", cache_dir=cache)
    assert not cache.exists()


def test_partial_model_write_cleans_only_its_own_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    existing = cache / "existing.png"
    existing.write_bytes(b"existing snapshot")
    open_path = Path.open
    cleanup_targets: list[Path] = []

    class FailingWriter:
        """写入部分模型数据后模拟磁盘失败。"""

        def __init__(self, stream: BinaryIO) -> None:
            self.stream = stream

        def __enter__(self) -> FailingWriter:
            return self

        def __exit__(self, *args: Any) -> None:
            self.stream.close()

        def write(self, data: bytes) -> int:
            self.stream.write(data[:8])
            raise OSError("model write interrupted")

    def fail_model_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        stream = open_path(path, mode, *args, **kwargs)
        return FailingWriter(stream) if mode == "xb" and ".model." in path.name else stream

    def record_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        cleanup_targets.append(path)

    monkeypatch.setattr(Path, "open", fail_model_open)
    monkeypatch.setattr(Path, "unlink", record_cleanup)
    with pytest.raises(IrisImageError, match="保存图片副本失败"):
        save_image(_image_bytes(image_format="JPEG", orientation=6), cache_dir=cache)
    assert existing.read_bytes() == b"existing snapshot"
    assert len(cleanup_targets) == 2
    assert {path.name.split(".")[1] for path in cleanup_targets} == {"original", "model"}
    assert set(cleanup_targets) == set(cache.iterdir()) - {existing}
