"""加载已发布概览，在窗口采用边界选择核心事实与知识范围或仅知识范围。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..exceptions import IrisContextError
from ..lifecycle import MemoryOverviewSource, SessionContextWindow
from ..memory import MemoryService
from ..message import LLMRequest
from ..providers.protocols import CompletionProvider


async def load_context_windows(
    *,
    memory_service: MemoryService | None,
    namespaces: Sequence[str],
    tool_names: Sequence[str],
) -> tuple[SessionContextWindow, SessionContextWindow]:
    """一次读取发布物，构造共享来源的 full 和 navigation 候选。"""
    if memory_service is None or not namespaces:
        empty = SessionContextWindow()
        return empty, empty
    documents = await memory_service.aload_overviews(namespaces)

    instructions = (
        "以当前概览为长期记忆范围；没有提及的主题默认没有，不搜索这些主题。"
        "仅对概览已覆盖且问题需要的主题按需读取，无关问题无需读取。"
        "没有概览则本窗口暂不使用长期记忆，正常聊天但不查询长期记忆。"
        "概览或文件未同步的提示不扩展主题范围，也不阻断已覆盖主题的数据库查询；"
        "概览可能旧于数据库当前记录，不能视为已核实的当前值。"
    )
    if "memory_search" in tool_names:
        instructions += (
            " 可用 memory_search 以普通文本关键词检索，可选 categories/kinds。"
            "结果只返回候选，has_more 时收紧查询，is_complete 只表示正文是否完整；"
            "片段充分时可直接使用。"
        )
    if "memory_fetch" in tool_names:
        instructions += (
            " 可用 memory_fetch 按已知 item_id 读取当前完整记录；"
            "需要完整记录或来源时再按需读取。"
        )
    if "memory_search" not in tool_names and "memory_fetch" not in tool_names:
        instructions += " 当前没有专用数据库读取工具。"

    sources = tuple(
        MemoryOverviewSource.model_construct(
            namespace=document.namespace,
            path=document.path.as_posix(),
            source_revision=document.source_revision,
        )
        for document in documents
    )
    full_parts = ["# Memory overview", instructions]
    navigation_parts = [
        "# Memory overview",
        "本窗口仅载入知识范围，未载入核心事实。",
        instructions,
    ]
    for document in documents:
        heading = f"## {document.namespace}"
        warning = f"\n\n{document.warning}" if document.warning else ""
        full_parts.append(f"{heading}{warning}\n\n{document.text}")
        navigation_parts.append(f"{heading}{warning}\n\n{document.navigation}")
    if not documents:
        navigation_parts.extend(
            f"## {namespace}\n\n未配置概览发布物，本窗口不使用长期记忆。"
            for namespace in namespaces
        )
    navigation = SessionContextWindow.model_construct(
        memory_overview="\n\n".join(navigation_parts), mode="navigation", sources=sources
    )
    if all(document.source_revision is None for document in documents):
        return navigation, navigation
    full = SessionContextWindow.model_construct(
        memory_overview="\n\n".join(full_parts), mode="full", sources=sources
    )
    return full, navigation


def select_context_window(
    *,
    candidates: tuple[SessionContextWindow, SessionContextWindow],
    build_request: Callable[[SessionContextWindow], LLMRequest],
    provider: CompletionProvider,
    memory_budget_tokens: int,
    input_budget_tokens: int,
) -> tuple[SessionContextWindow, LLMRequest]:
    """以实际完整请求选择窗口，专用额度只计算 addendum 带来的输入增量。"""
    full, navigation = candidates
    base_request = build_request(SessionContextWindow())
    base_tokens = provider.estimate_input_tokens(base_request)
    try:
        full_request = build_request(full)
    except IrisContextError as exc:
        # 只有完整 system 的字符容量错误可以通过切换知识范围解决。
        if exc.context.get("section") != "system" or "limit" not in exc.context:
            raise
    else:
        full_tokens = provider.estimate_input_tokens(full_request)
        if full_tokens - base_tokens <= memory_budget_tokens and full_tokens <= input_budget_tokens:
            return full, full_request

    navigation_request = build_request(navigation)
    navigation_tokens = provider.estimate_input_tokens(navigation_request) - base_tokens
    if navigation_tokens > memory_budget_tokens:
        raise IrisContextError(
            "memory 完整知识范围超过窗口预算",
            section="memory_overview",
            actual=navigation_tokens,
            limit=memory_budget_tokens,
        )
    # 完整请求中的既有历史仍由正常compaction处理，不在这里重复实现切点与容量错误。
    return navigation, navigation_request


__all__ = ["load_context_windows", "select_context_window"]
