"""将显式召回结果转换为独立的动态记忆历史消息。"""

from __future__ import annotations

from ..context import ContextBuilder, ContextSection, ContextSlot
from ..exceptions import IrisMemoryError
from ..lifecycle import RuntimeExecutionOptions
from ..memory import (
    MemoryContextBuilder,
    MemoryQuery,
    MemorySearchResult,
    MemoryService,
)
from ..message import Msg


async def prepare_run_memory_messages(
    *,
    options: RuntimeExecutionOptions,
    memory_service: MemoryService | None,
    memory_context_builder: MemoryContextBuilder,
    context_builder: ContextBuilder,
) -> tuple[Msg, ...]:
    """在输入提交前读取一次显式 memory，渲染供历史重放的实际片段。"""
    if options.memory_results is None and options.memory_query is None:
        return ()

    if options.memory_results is not None:
        results = [MemorySearchResult.model_validate(item) for item in options.memory_results]
        bundle = memory_context_builder.build(
            results,
            max_chars=options.memory_max_chars,
        )
    else:
        if memory_service is None:
            raise IrisMemoryError("显式 memory_query 需要注入 memory_service")
        bundle = await memory_service.abuild_context(
            MemoryQuery.model_validate(options.memory_query),
            max_chars=options.memory_max_chars,
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
