"""工具侧图片导入：确定 session 目录并复用共享图片处理。"""

from pathlib import Path

from ..message import ImageBlock, image_block_from_saved
from ..utils.images import save_image
from ._paths import safe_path_segment
from .base import ToolExecutionContext


def import_tool_image(
    source: Path | bytes, context: ToolExecutionContext, *, name: str | None = None
) -> ImageBlock:
    """为工具结果准备图片引用，由异步调用方安排本地 I/O 执行位置。

    Args:
        source: 原始图片 bytes 或路径，相对路径以工具 workspace 解析。
        context: 提供 workspace 与 session 身份的已准备执行上下文。
        name: 可选图片显示名称。

    Returns:
        完整保存后交付的图片块；合规缓存源可直接复用。

    Raises:
        IrisImageError: 图片读取、解码、处理或保存失败。
    """
    cache_root = (context.workspace_root / ".iris" / "image-cache").resolve()
    resolved = (context.workspace_root / source).resolve() if isinstance(source, Path) else source
    saved = save_image(
        resolved,
        cache_dir=cache_root / safe_path_segment(context.session_id),
        reuse_source=isinstance(resolved, Path) and resolved.is_relative_to(cache_root),
    )
    return image_block_from_saved(saved, name=name)


__all__ = ["import_tool_image"]
