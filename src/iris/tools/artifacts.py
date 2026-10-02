"""工具结果正文与显式文件产物的本地存储。

用于处理执行结果体积过大时的内容截断与外部文件持久化机制。
如果输出短，直接返回原始结果；如果输出长，则自动将原始内容存入隐藏工作区，
向 LLM 返回提示信息和有界正文预览，避免 token 超限。
显式发布文件时保存独立二进制副本，并返回供宿主使用的产物引用。

Example:
    store = ToolArtifactStore(Path(".iris/tool-results"))
"""

# region imports
from __future__ import annotations

import json
import mimetypes
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from ..exceptions import IrisToolExecutionError
from ..message import DataBlock, ImageBlock, TextBlock, image_reference_text
from ._paths import safe_path_segment
from .base import ToolArtifact, ToolExecutionContext, ToolResult

# endregion


class ToolArtifactStore:
    """将工具结果正文和发布文件副本写入 `.iris/tool-results`。

    在长内容导致 LLM 无法容纳上下文时，自动提取负载并放入文件中，原位放置小尺寸报告文件。

    Attributes:
        root (Path): 存放结果文件的根目录，通常为 `.iris/tool-results/{session_id}`。
        preview_chars (int): 截断后向大模型展示的正文预览字符长度。
        preview_mode (Literal["head", "head_tail"]): 正文预览保留前缀或头尾。

    Example:
        store = ToolArtifactStore(Path(".iris/tool-results"))
        res = store.persist_if_large(tool_result, max_chars=10000)
    """

    def __init__(
        self,
        root: Path,
        preview_chars: int = 8000,
        *,
        preview_mode: Literal["head", "head_tail"] = "head",
    ) -> None:
        """初始化 artifact 存储目录。

        建立自动截断存储策略的依赖注入与常规限制设定。

        Args:
            root (Path): 写入目标的基础系统路径。
            preview_chars (int): 缩略内容预览字符数量设定。
            preview_mode: 正文预览布局。

        Returns:
            None
        """
        self.root = root
        self.preview_chars = preview_chars
        self.preview_mode = preview_mode

    def persist_file(self, tool_use_id: str, source: Path, *, preview: str) -> ToolArtifact:
        """分块复制已解析源文件，完整关闭后交付独立副本；失败删除本次半成品。"""
        created = False
        try:
            path = self._new_path(tool_use_id, source.suffix)
            with source.open("rb") as original, path.open("xb") as output:
                created = True
                shutil.copyfileobj(original, output, length=1024 * 1024)
                size = output.tell()
            mime_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        except (OSError, ValueError) as exc:
            if created:
                try:
                    path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    raise IrisToolExecutionError(
                        "ARTIFACT_ERROR: 复制失败且部分副本无法删除"
                    ) from cleanup_error
            raise IrisToolExecutionError("ARTIFACT_ERROR: 复制发布文件失败") from exc
        return ToolArtifact(path=path, mime_type=mime_type, size_bytes=size, preview=preview)

    def _new_path(self, tool_use_id: str, suffix: str) -> Path:
        """统一生成当前 store 下尚未写入的唯一结果路径。"""
        root = self.root.resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        path = (root / f"{safe_path_segment(tool_use_id)}-{uuid4().hex}{suffix}").resolve(
            strict=False
        )
        path.relative_to(root)
        return path

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
            path = self._new_path(tool_use_id, suffix)
            with path.open("xb") as output:
                size = output.write(content.encode("utf-8"))
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
        preview = _preview_text(content, self.preview_chars, self.preview_mode)
        image_refs = "\n".join(
            image_reference_text(block) for block in result.content if isinstance(block, ImageBlock)
        )
        saved_content = f"{image_refs}\n\n{content}" if image_refs else content
        text_artifact = self._persist_text(
            result.tool_use_id,
            saved_content,
            suffix=".model.txt" if result.artifact is not None else ".txt",
            mime_type="text/plain",
            preview=preview,
        )
        artifact = (result.artifact or text_artifact).model_copy(
            update={"text_path": text_artifact.path}
        )

        # --- 3. Replace memory text with preview ---
        suffix = (
            f"\n\n[工具 {result.tool_name} 结果已截断，已保存的工具结果正文：{text_artifact.path}。"
            "可读取该文件查看已保存正文。建议将 .iris/ 加入 .gitignore。]"
        )
        if result.artifact is not None:
            suffix += f"\n[原生结果：{artifact.path}]"
        prefix_chars = (
            len(f"Error[{result.error.code}]: ") if result.is_error and result.error else 0
        )
        if len(suffix) + prefix_chars > max_chars:
            raise IrisToolExecutionError("ARTIFACT_ERROR: 工具结果额度不足以容纳完整回读提示")
        limited = truncate_tool_result(
            result,
            max_chars=max_chars,
            preview_chars=self.preview_chars,
            suffix=suffix,
            preview_mode=self.preview_mode,
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
    result: ToolResult,
    *,
    max_chars: int,
    preview_chars: int,
    suffix: str,
    preview_mode: Literal["head", "head_tail"] = "head",
) -> ToolResult:
    """按最终正文预算裁剪，保留错误码和调用方提供的完整取回提示。

    预算不足提示长度时仅保留提示及错误前缀，不创建文件。

    Args:
        result: 待裁剪的工具结果。
        max_chars: 包含错误前缀和提示的目标字符预算。
        preview_chars: 正文预览的最大字符数。
        suffix: 必须保留的取回或截断说明。
        preview_mode: 正文预览布局。

    Returns:
        已同步正文与错误说明的工具结果。
    """
    error = result.error if result.is_error else None
    prefix_chars = len(f"Error[{error.code}]: ") if error else 0
    body = result.model_content[prefix_chars:]
    available = max(0, max_chars - len(suffix) - prefix_chars)
    message = _preview_text(body, min(preview_chars, available), preview_mode) + suffix
    content: list[DataBlock] = []
    text_replaced = False
    for block in result.content:
        if isinstance(block, ImageBlock):
            content.append(block)
        elif not text_replaced:
            content.append(TextBlock(text=message))
            text_replaced = True
    if not text_replaced:
        content.insert(0, TextBlock(text=message))
    return result.model_copy(
        update={
            "content": content,
            "error": error.model_copy(update={"message": message}) if error else result.error,
            "hook_feedback": (),
        }
    )


def _preview_text(text: str, limit: int, mode: Literal["head", "head_tail"]) -> str:
    """只在实际裁剪时插入省略标记，标记占用同一字符额度。"""
    if len(text) <= limit or mode == "head":
        return text[:limit]
    marker = "\n[中间内容已省略]\n"
    if limit <= len(marker):
        return marker[:limit]
    available = limit - len(marker)
    head = (available + 1) // 2
    tail = available - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def artifact_store_for(
    context: ToolExecutionContext,
    *,
    preview_chars: int,
    preview_mode: Literal["head", "head_tail"] = "head",
) -> ToolArtifactStore:
    """按本次调用的 session 取得既有 artifact store，不绑定某个 root run。"""
    session_id = safe_path_segment(context.session_id)
    root = context.workspace_root / ".iris" / "tool-results" / session_id
    return ToolArtifactStore(root=root, preview_chars=preview_chars, preview_mode=preview_mode)
