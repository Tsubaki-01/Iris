"""Runtime memory 适配层。

本模块只把显式 memory 输入转换为 `iris.context` slots，不持有召回、存储、
provider 或 session 逻辑。
"""

from __future__ import annotations

from ..context import ContextBuildInput, ContextSlot
from ..exceptions import IrisMemoryError
from ..lifecycle import RuntimeExecutionOptions
from ..memory import (
    MemoryContextBuilder,
    MemoryContextBundle,
    MemoryQuery,
    MemorySearchResult,
    MemoryService,
)


async def prepare_activation_memory_context_input(
    context_input: ContextBuildInput,
    *,
    options: RuntimeExecutionOptions,
    memory_service: MemoryService | None,
    memory_context_builder: MemoryContextBuilder,
) -> ContextBuildInput:
    """把 lifecycle JSON snapshot 转回现有 memory 构建边界。"""
    if options.memory_results is None and options.memory_query is None:
        return context_input

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

    return context_input.with_memory_slots(*_memory_bundle_to_slots(bundle))


def _memory_bundle_to_slots(bundle: MemoryContextBundle) -> list[ContextSlot]:
    """将 memory bundle 转换成 prompt-facing context slots。"""
    slots: list[ContextSlot] = []
    for fragment in bundle.fragments:
        slots.append(
            ContextSlot(
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
        )
    return slots


__all__ = ["prepare_activation_memory_context_input"]
