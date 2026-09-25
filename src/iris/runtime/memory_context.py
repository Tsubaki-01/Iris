"""加载已发布概览，在窗口采用边界选择核心事实与知识范围或仅知识范围。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..exceptions import IrisContextError
from ..lifecycle import MemoryOverviewSource, SessionContextWindow
from ..memory import MemoryService
from ..message import LLMRequest
from ..providers.protocols import CompletionProvider
from ..utils import TemplateRenderer
from ._prompts import render_prompt

_MEMORY_CONTEXT_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "memory_context.j2"


async def load_context_windows(
    *,
    prompt_renderer: TemplateRenderer,
    memory_service: MemoryService | None,
    namespaces: Sequence[str],
    tool_names: Sequence[str],
) -> tuple[SessionContextWindow, SessionContextWindow]:
    """一次读取发布物，构造共享来源的 full 和 navigation 候选。"""
    if memory_service is None or not namespaces:
        empty = SessionContextWindow()
        return empty, empty
    documents = await memory_service.aload_overviews(namespaces)

    sources = tuple(
        MemoryOverviewSource.model_construct(
            namespace=document.namespace,
            path=document.path.as_posix(),
            source_revision=document.source_revision,
        )
        for document in documents
    )
    template_context: dict[str, Any] = {
        "documents": documents,
        "namespaces": namespaces,
        "has_memory_search": "memory_search" in tool_names,
        "has_memory_fetch": "memory_fetch" in tool_names,
    }
    navigation = SessionContextWindow.model_construct(
        memory_overview=render_prompt(
            prompt_renderer, _MEMORY_CONTEXT_PROMPT, {**template_context, "mode": "navigation"}
        ),
        mode="navigation",
        sources=sources,
    )
    if all(document.source_revision is None for document in documents):
        return navigation, navigation
    full = SessionContextWindow.model_construct(
        memory_overview=render_prompt(
            prompt_renderer, _MEMORY_CONTEXT_PROMPT, {**template_context, "mode": "full"}
        ),
        mode="full",
        sources=sources,
    )
    return full, navigation


def select_context_window(
    *,
    candidates: tuple[SessionContextWindow, SessionContextWindow],
    build_request: Callable[[SessionContextWindow], LLMRequest],
    provider: CompletionProvider,
    memory_budget_tokens: int,
    input_budget_tokens: int,
) -> tuple[SessionContextWindow, LLMRequest, int]:
    """返回窗口、选定请求及完整 token 数；专用额度只计算 addendum 的增量。"""
    full, navigation = candidates
    base_request = build_request(SessionContextWindow())
    base_tokens = provider.estimate_input_tokens(base_request)
    navigation_request: LLMRequest | None = None
    try:
        full_request = build_request(full)
    except IrisContextError as exc:
        # 只有完整 system 的字符容量错误可以通过切换知识范围解决。
        if (
            exc.context.get("section") != "system"
            or "limit" not in exc.context
            or full is navigation
        ):
            raise
    else:
        full_tokens = provider.estimate_input_tokens(full_request)
        if full_tokens - base_tokens <= memory_budget_tokens and full_tokens <= input_budget_tokens:
            return full, full_request, full_tokens
        if full is navigation:
            navigation_request, navigation_total = full_request, full_tokens
    if navigation_request is None:
        navigation_request = build_request(navigation)
        navigation_total = provider.estimate_input_tokens(navigation_request)

    navigation_tokens = navigation_total - base_tokens
    if navigation_tokens > memory_budget_tokens:
        raise IrisContextError(
            "memory 完整知识范围超过窗口预算",
            section="memory_overview",
            actual=navigation_tokens,
            limit=memory_budget_tokens,
        )
    # 完整请求中的既有历史仍由正常compaction处理，不在这里重复实现切点与容量错误。
    return navigation, navigation_request, navigation_total


__all__ = ["load_context_windows", "select_context_window"]
