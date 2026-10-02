"""将已保存的模型图片编码为本次请求内容，不处理或修改图片。"""

from __future__ import annotations

import base64

from ..exceptions import IrisImageError
from ..message import ImageBlock


def image_data_url(block: ImageBlock) -> str:
    """读取模型副本并编码，文件信息沿用导入时的可信结果。

    Args:
        block: 指向已完整保存的图片副本的内容块。

    Returns:
        具有实际模型版 MIME 的 data URL。

    Raises:
        IrisImageError: 模型副本无法读取。
    """
    try:
        data = block.model.path.read_bytes()
    except OSError as exc:
        raise IrisImageError("无法读取模型图片副本", path=str(block.model.path)) from exc
    return f"data:{block.model.mime_type};base64,{base64.b64encode(data).decode('ascii')}"
