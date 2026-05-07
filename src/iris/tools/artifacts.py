"""工具大结果 artifact 存储。"""

from __future__ import annotations

import re
from pathlib import Path

from ..exceptions import IrisToolExecutionError
from ..message import TextBlock
from .base import ToolArtifact, ToolResult


class ToolArtifactStore:
    """将超大工具结果写入 `.iris/tool-results`。"""

    def __init__(self, root: Path, preview_chars: int = 8000) -> None:
        """初始化 artifact 存储目录。"""
        self.root = root
        self.preview_chars = preview_chars

    def persist_if_large(self, result: ToolResult, *, max_chars: int) -> ToolResult:
        """必要时将工具结果落盘，并把模型内容替换为预览说明。"""
        content = result.model_content()
        if result.is_error or len(content) <= max_chars:
            return result
        try:
            root = self.root.resolve(strict=False)
            root.mkdir(parents=True, exist_ok=True)
            artifact_path = (root / f"{_safe_path_segment(result.tool_use_id)}.txt").resolve(
                strict=False
            )
            artifact_path.relative_to(root)
            artifact_path.write_text(content, encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise IrisToolExecutionError("ARTIFACT_ERROR: 写入工具 artifact 失败") from exc
        preview = content[: self.preview_chars]
        stat = artifact_path.stat()
        artifact = ToolArtifact(
            path=artifact_path,
            mime_type="text/plain",
            size_bytes=stat.st_size,
            preview=preview,
        )
        message = (
            f"{preview}\n\n"
            f"[结果已截断，完整内容已写入 {artifact_path}，大小 {stat.st_size} bytes。"
            " 可使用 read_file 读取该路径。建议将 .iris/ 加入 .gitignore。]"
        )
        return result.model_copy(
            update={
                "content": [TextBlock(text=message)],
                "artifact": artifact,
                "metadata": {
                    **result.metadata,
                    "gitignore_hint": "建议将 .iris/ 加入 .gitignore",
                },
            }
        )


def _safe_path_segment(value: str) -> str:
    """将外部 ID 转为单个安全路径段。"""
    segment = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    return segment.strip("_") or "default"
