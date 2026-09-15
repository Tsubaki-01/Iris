"""工具大结果 artifact 存储。

用于处理执行结果体积过大时的内容截断与外部文件持久化机制。
如果输出短，直接返回原始结果；如果输出长，则自动将原始内容存入隐藏工作区，
向 LLM 返回一个包含提示信息和部分前缀短预览的替换结果，避免 token 超限。

Example:
    store = ToolArtifactStore(Path(".iris/tool-results"))
"""

# region imports
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..exceptions import IrisToolExecutionError
from ..message import TextBlock
from ._paths import safe_path_segment
from .base import ToolArtifact, ToolExecutionContext, ToolResult

# endregion


class ToolArtifactStore:
    """将超大工具结果写入 `.iris/tool-results`。

    在长内容导致 LLM 无法容纳上下文时，自动提取负载并放入文件中，原位放置小尺寸报告文件。

    Attributes:
        root (Path): 存放结果文件的根目录，通常为 `.iris/tool-results/{session_id}`。
        preview_chars (int): 截断后向大模型展示的文件头部预览字符长度。

    Example:
        store = ToolArtifactStore(Path(".iris/tool-results"))
        res = store.persist_if_large(tool_result, max_chars=10000)
    """

    def __init__(self, root: Path, preview_chars: int = 8000) -> None:
        """初始化 artifact 存储目录。

        建立自动截断存储策略的依赖注入与常规限制设定。

        Args:
            root (Path): 写入目标的基础系统路径。
            preview_chars (int): 缩略内容预览字符数量设定。

        Returns:
            None
        """
        self.root = root
        self.preview_chars = preview_chars

    def persist_json(
        self,
        tool_use_id: str,
        payload: Mapping[str, Any],
        *,
        preview: str,
    ) -> ToolArtifact:
        """严格序列化完整 MCP 结果；序列化或落盘失败统一报告 artifact 错误。"""
        try:
            content = json.dumps(dict(payload), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise IrisToolExecutionError("ARTIFACT_ERROR: MCP 结果无法序列化") from exc
        return self._persist_text(
            tool_use_id, content, suffix=".mcp.json", mime_type="application/json", preview=preview
        )

    def _persist_text(
        self,
        tool_use_id: str,
        content: str,
        *,
        suffix: str,
        mime_type: str,
        preview: str,
    ) -> ToolArtifact:
        """复用单一路径编码和写入边界。"""
        try:
            root = self.root.resolve(strict=False)
            root.mkdir(parents=True, exist_ok=True)
            path = (root / f"{safe_path_segment(tool_use_id)}{suffix}").resolve(strict=False)
            path.relative_to(root)
            size = path.write_bytes(content.encode("utf-8"))
        except (OSError, ValueError) as exc:
            raise IrisToolExecutionError("ARTIFACT_ERROR: 写入工具 artifact 失败") from exc
        return ToolArtifact(path=path, mime_type=mime_type, size_bytes=size, preview=preview)

    def persist_if_large(
        self,
        result: ToolResult,
        *,
        max_chars: int,
    ) -> ToolResult:
        """必要时将工具结果落盘，并把模型内容替换为预览说明。

        为了防御异常长文本输出对代理内存产生的压垮效应，
        接管返回对象并转写为本地盘文件，用轻量级的提示对象换出沉重的文本块。

        Args:
            result (ToolResult): 原始被挂起审核容量安全性的执行包。
            max_chars (int): 阈值门限数值，越过即触发外存交换。

        Returns:
            ToolResult: 未超时原文结构或带有截断声明字样与落盘路径元数据的新结果实体。

        Raises:
            IrisToolExecutionError: 读写文件权限不足或路径不可达时向上冒出文件挂载异常。

        Example:
            >>> small_result = ToolResult(
            ...     tool_use_id="1",
            ...     tool_name="x",
            ...     content=[TextBlock(text="a")],
            ... )
            >>> store.persist_if_large(small_result, max_chars=100)
            [ToolResult keeps original content]
        """
        # --- 1. Evaluate content length threshold ---
        content = result.model_content
        if len(content) <= max_chars:
            return result

        # --- 2. Write artifact payload to disk ---
        preview = content[: self.preview_chars]
        artifact = result.artifact or self._persist_text(
            result.tool_use_id,
            content,
            suffix=".txt",
            mime_type="text/plain",
            preview=preview,
        )

        # --- 3. Replace memory text with preview ---
        suffix = (
            f"\n\n[结果已截断，完整内容已写入 {artifact.path}，大小 {artifact.size_bytes} bytes。"
            " 可使用 read_file 读取该路径。建议将 .iris/ 加入 .gitignore。]"
        )
        limited = truncate_tool_result(
            result, max_chars=max_chars, preview_chars=self.preview_chars, suffix=suffix
        )
        return limited.model_copy(
            update={
                "artifact": artifact,
                "metadata": {
                    **result.metadata,
                    "gitignore_hint": "建议将 .iris/ 加入 .gitignore",
                },
            }
        )


def truncate_tool_result(
    result: ToolResult, *, max_chars: int, preview_chars: int, suffix: str
) -> ToolResult:
    """按最终正文预算裁剪，保留错误码和调用方提供的完整取回提示。

    预算不足提示长度时仅保留提示及错误前缀，不创建文件。

    Args:
        result: 待裁剪的工具结果。
        max_chars: 包含错误前缀和提示的目标字符预算。
        preview_chars: 正文预览的最大字符数。
        suffix: 必须保留的取回或截断说明。

    Returns:
        已同步正文与错误说明的工具结果。
    """
    error = result.error if result.is_error else None
    prefix_chars = len(f"Error[{error.code}]: ") if error else 0
    body = error.message if error else result.model_content
    available = max(0, max_chars - len(suffix) - prefix_chars)
    message = body[: min(preview_chars, available)] + suffix
    return result.model_copy(
        update={
            "content": [TextBlock(text=message)],
            "error": error.model_copy(update={"message": message}) if error else result.error,
        }
    )


def artifact_store_for(context: ToolExecutionContext, *, preview_chars: int) -> ToolArtifactStore:
    """按本次调用的 session 取得既有 artifact store，不绑定某个 root run。"""
    session_id = safe_path_segment(context.session_id)
    root = context.workspace_root / ".iris" / "tool-results" / session_id
    return ToolArtifactStore(root=root, preview_chars=preview_chars)
