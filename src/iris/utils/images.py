"""共享图片处理与文件副本保存；调用方提供明确的缓存目录。

本模块只返回字节和文件信息，不持有 session、消息或 provider 格式。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from ..exceptions import IrisImageError

MAX_IMAGE_DIMENSION = 2000
MAX_IMAGE_BYTES = 15 * 1024 * 1024 // 4
_MIME_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


@dataclass(frozen=True, slots=True)
class ImageData:
    """一份确定编码及尺寸的图片字节。"""

    data: bytes
    mime_type: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """原始字节与已满足模型处理策略的字节；无需变换时复用同一对象。"""

    original: ImageData
    model: ImageData


@dataclass(frozen=True, slots=True)
class SavedImageFile:
    """已完整关闭的图片文件及其实际格式、尺寸。"""

    path: Path
    mime_type: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class SavedImage:
    """已保存的原图和模型版；无需变换时指向同一文件。"""

    original: SavedImageFile
    model: SavedImageFile


def prepare_image(data: bytes) -> PreparedImage:
    """一次解码并按固定策略准备静态 PNG、JPEG 或 WebP。

    Args:
        data: 原始图片编码；格式从内容取得。

    Returns:
        保持原字节的 original 与有界模型版，不生成 base64。

    Raises:
        IrisImageError: 格式不支持、解码失败或有限处理仍超过字节上限。
    """
    try:
        with Image.open(BytesIO(data)) as decoded:
            image_format = decoded.format
            if image_format not in _MIME_TYPES or getattr(decoded, "n_frames", 1) != 1:
                raise IrisImageError("仅支持静态 PNG、JPEG 和 WebP 图片")
            decoded.load()
            original = ImageData(data, _MIME_TYPES[image_format], *decoded.size)
            needs_orientation = decoded.getexif().get(274) in {2, 3, 4, 5, 6, 7, 8}
            if (
                max(decoded.size) <= MAX_IMAGE_DIMENSION
                and len(data) <= MAX_IMAGE_BYTES
                and not needs_orientation
            ):
                return PreparedImage(original, original)

            has_alpha = "A" in decoded.getbands() or "transparency" in decoded.info
            with ImageOps.exif_transpose(decoded) as oriented:
                with oriented.convert("RGBA" if has_alpha else "RGB") as pixels:
                    scale = min(1.0, MAX_IMAGE_DIMENSION / max(pixels.size))
                    width = max(1, int(pixels.width * scale))
                    height = max(1, int(pixels.height * scale))
                    for reduction in range(3):
                        size = (max(1, width // 2**reduction), max(1, height // 2**reduction))
                        with pixels.resize(size, Image.Resampling.LANCZOS) as candidate:
                            if image_format == "PNG" or has_alpha:
                                model = _encode(candidate, "PNG")
                                if len(model.data) <= MAX_IMAGE_BYTES:
                                    return PreparedImage(original, model)
                            if not has_alpha:
                                for quality in (85, 70, 50, 30):
                                    model = _encode(candidate, "JPEG", quality=quality)
                                    if len(model.data) <= MAX_IMAGE_BYTES:
                                        return PreparedImage(original, model)
    except (OSError, ValueError) as exc:
        raise IrisImageError("图片解码或处理失败") from exc
    raise IrisImageError("有限缩放和编码后图片仍超过模型版字节上限", max_bytes=MAX_IMAGE_BYTES)


def _encode(image: Image.Image, image_format: str, *, quality: int | None = None) -> ImageData:
    """从同一解码图的当前尺寸生成一个候选，不重复压缩有损结果。"""
    output = BytesIO()
    options = {"quality": quality} if quality is not None else {}
    image.save(output, format=image_format, optimize=True, **options)
    return ImageData(output.getvalue(), _MIME_TYPES[image_format], *image.size)


def save_image(source: Path | bytes, *, cache_dir: Path) -> SavedImage:
    """保存一次导入快照，两个文件完整关闭后才交付引用。

    Args:
        source: 已由调用方解析的文件路径，或原始图片 bytes。
        cache_dir: 调用方确定的目标目录；本函数不知道 session。

    Returns:
        绝对路径形式的原图和模型版文件信息。

    Raises:
        IrisImageError: 读取、处理或写入失败；只清理本次未完成副本。
    """
    try:
        if isinstance(source, Path):
            buffer = BytesIO()
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, buffer, length=1024 * 1024)
            data = buffer.getvalue()
        else:
            data = source
        prepared = prepare_image(data)
        directory = cache_dir.resolve()
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise IrisImageError("读取图片或创建副本目录失败") from exc

    asset_id = uuid4().hex
    original = _file_info(directory, asset_id, "original", prepared.original)
    model = (
        original
        if prepared.model is prepared.original
        else _file_info(directory, asset_id, "model", prepared.model)
    )
    pending = [(original.path, prepared.original.data)]
    if model is not original:
        pending.append((model.path, prepared.model.data))
    created: list[Path] = []
    try:
        for path, content in pending:
            with path.open("xb") as output:
                created.append(path)
                shutil.copyfileobj(BytesIO(content), output, length=1024 * 1024)
    except OSError as exc:
        try:
            for path in created:
                path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            raise IrisImageError("图片保存失败且本次半成品无法清理") from cleanup_error
        raise IrisImageError("保存图片副本失败") from exc
    return SavedImage(original, model)


def _file_info(directory: Path, asset_id: str, role: str, image: ImageData) -> SavedImageFile:
    """按实际输出格式构造同一随机资产 ID 下的文件名。"""
    return SavedImageFile(
        directory / f"{asset_id}.{role}{_EXTENSIONS[image.mime_type]}",
        image.mime_type,
        image.width,
        image.height,
    )


__all__ = [
    "ImageData",
    "PreparedImage",
    "SavedImageFile",
    "SavedImage",
    "prepare_image",
    "save_image",
]
