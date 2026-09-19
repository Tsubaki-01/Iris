"""选择本轮记忆来源，渲染并去重供历史重放的动态快照。"""

from __future__ import annotations

import logging

from ..context import ContextBuilder, ContextSection, ContextSlot
from ..exceptions import IrisMemoryError
from ..lifecycle import RuntimeExecutionOptions
from ..memory import (
    MemoryConfig,
    MemoryContextBuilder,
    MemoryQuery,
    MemorySearchResult,
    MemoryService,
)
from ..message import Msg

logger = logging.getLogger(__name__)


async def prepare_run_memory_messages(
    *,
    options: RuntimeExecutionOptions,
    memory_service: MemoryService | None,
    memory_context_builder: MemoryContextBuilder,
    context_builder: ContextBuilder,
    config: MemoryConfig,
    run_input: str,
    run_id: str,
    visible_history: list[Msg],
) -> tuple[Msg, ...]:
    """显式输入优先；自动召回只过滤当前可见的同 ID、同渲染片段。"""
    automatic = False
    if options.memory_results is not None:
        results = [MemorySearchResult.model_validate(item) for item in options.memory_results]
        bundle = memory_context_builder.build(
            results,
            max_chars=options.memory_max_chars,
        )
    elif options.memory_query is not None:
        if memory_service is None:
            raise IrisMemoryError("显式 memory_query 需要注入 memory_service")
        bundle = await memory_service.abuild_context(
            MemoryQuery.model_validate(options.memory_query),
            max_chars=options.memory_max_chars,
        )
    elif memory_service is not None and config.recall_mode == "on_turn":
        automatic = True
        query = MemoryQuery.model_construct(
            namespaces=config.read_namespaces,
            text=run_input,
            max_query_terms=config.max_query_terms,
        )
        try:
            results = await memory_service.arecall(query)
        except Exception:
            logger.warning("自动 memory 读取失败，继续本轮对话 run_id=%s", run_id, exc_info=True)
            return ()
        bundle = memory_context_builder.build(results, max_chars=options.memory_max_chars)
    else:
        return ()

    visible_fragments = (
        {
            (message.metadata["item_id"], message.text)
            for message in visible_history
            if message.metadata.get("context_kind") == "memory"
        }
        if automatic
        else set()
    )

    messages: list[Msg] = []
    for fragment in bundle.fragments:
        slot = ContextSlot.model_construct(
            name="memory",
            content=fragment.text,
            attributes={
                "item_id": fragment.item_id,
                "category": fragment.category.value,
                "kind": fragment.kind.value,
                "level": fragment.level.value,
                "truncated": str(fragment.truncated).lower(),
            },
        )
        text = context_builder.render_section(
            "memory", ContextSection.model_construct(slots=[slot])
        )
        if automatic and (fragment.item_id, text) in visible_fragments:
            continue
        messages.append(
            Msg.user(
                text,
                sender="context",
                metadata={
                    "context_kind": "memory",
                    "item_id": fragment.item_id,
                    "namespace": fragment.namespace,
                    "truncated": fragment.truncated,
                },
            )
        )
    return tuple(messages)


__all__ = ["prepare_run_memory_messages"]
