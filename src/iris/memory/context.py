"""记忆上下文构建器。

本模块只负责把召回结果裁剪成结构化片段，不拼接完整 prompt。

Example:
    bundle = MemoryContextBuilder().build(results, max_chars=4000)
"""

# region imports
from __future__ import annotations

from ..exceptions import IrisMemoryError
from .models import MemoryContextBundle, MemoryContextFragment, MemorySearchResult

# endregion

# ==========================================
#                 Constants
# ==========================================
# region constants
MEMORY_CONTEXT_WARNING = "记忆内容可能过期或不完整，使用前应结合当前任务核对。"
# endregion


class MemoryContextBuilder:
    """将召回结果转换为有预算边界的记忆上下文。"""

    def build(
        self,
        results: list[MemorySearchResult],
        *,
        max_chars: int,
    ) -> MemoryContextBundle:
        """按顺序构建有字符预算的上下文片段。

        Args:
            results: 已按相关性或时间排序的召回结果。
            max_chars: 允许进入上下文的正文总字符数，必须为正数。

        Returns:
            MemoryContextBundle: 结构化记忆上下文。

        Raises:
            IrisMemoryError: 当字符预算不是正数时抛出。
        """
        if max_chars <= 0:
            raise IrisMemoryError(
                "memory context max_chars 必须为正数", max_chars=max_chars
            )

        fragments: list[MemoryContextFragment] = []
        total_chars = 0
        omitted_count = 0
        for index, result in enumerate(results):
            text = result.item.text
            next_total = total_chars + len(text)
            if next_total <= max_chars:
                fragments.append(
                    _fragment_from_result(result, text=text, truncated=False)
                )
                total_chars = next_total
                continue
            if not fragments:
                truncated_text = text[:max_chars]
                fragments.append(
                    _fragment_from_result(result, text=truncated_text, truncated=True)
                )
                total_chars = len(truncated_text)
                omitted_count = len(results) - index - 1
                break
            omitted_count = len(results) - index
            break

        return MemoryContextBundle(
            fragments=fragments,
            total_chars=total_chars,
            omitted_count=omitted_count,
            max_chars=max_chars,
        )


def _fragment_from_result(
    result: MemorySearchResult,
    *,
    text: str,
    truncated: bool,
) -> MemoryContextFragment:
    """从召回结果创建上下文片段。"""
    return MemoryContextFragment(
        item_id=result.item.id,
        text=text,
        category=result.item.category,
        kind=result.item.kind,
        level=result.item.level,
        reason=result.item.reason,
        confidence=result.item.confidence,
        importance=result.item.importance,
        warning=MEMORY_CONTEXT_WARNING,
        truncated=truncated,
    )
