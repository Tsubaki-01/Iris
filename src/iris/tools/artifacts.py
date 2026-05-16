"""工具大结果 artifact 存储。

用于处理执行结果体积过大时的内容截断与外部文件持久化机制。
如果输出短，直接返回原始结果；如果输出长，则自动将原始内容存入隐藏工作区，
向 LLM 返回一个包含提示信息和部分前缀短预览的替换结果，避免 token 超限。

Example:
    store = ToolArtifactStore(Path(".iris/tool-results"))
"""

# region imports
from __future__ import annotations

import re
from pathlib import Path

from ..exceptions import IrisToolExecutionError
from ..message import TextBlock
from .base import ToolArtifact, ToolResult

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

    def persist_if_large(self, result: ToolResult, *, max_chars: int) -> ToolResult:
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
        if result.is_error or len(content) <= max_chars:
            return result

        # --- 2. Write artifact payload to disk ---
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

        # --- 3. Replace memory text with preview ---
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
    """将外部 ID 转为单个安全路径段。

    抹除特异符号以规避外部系统调用注入针对系统路径操作解析器的越权注入行为。

    Args:
        value (str): 被校验整理替换的安全清洗上游字符段。

    Returns:
        str: 全部被置换为安全白名单内并剔除了多余头尾的文字串。
             若过滤后为空则返回 "default"。

    Example:
        >>> _safe_path_segment("../../foo!")
        "foo"
    """
    segment = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    return segment.strip("_") or "default"
