"""Runtime memory 适配层。

本模块只把显式 memory 输入转换为 `iris.context` slots，不持有召回、存储、
provider 或 session 逻辑。
"""

from __future__ import annotations

from ..context import ContextBuildInput, ContextSlot
from ..exceptions import IrisMemoryError
from ..memory import MemoryContextBuilder, MemoryContextBundle, MemoryService
from .models import RuntimeOptions


def prepare_memory_context_input(
    context_input: ContextBuildInput,
    *,
    options: RuntimeOptions,
    memory_service: MemoryService | None,
    memory_context_builder: MemoryContextBuilder,
) -> ContextBuildInput:
    """按显式 opt-in 选项追加运行态 memory slots。

    Args:
        context_input: 原始 context 构建输入。
        options: 本轮 runtime 选项。
        memory_service: 可选 memory 服务；仅 `memory_query` 路径需要。
        memory_context_builder: 用于把预先召回结果裁剪成 memory bundle。

    Returns:
        ContextBuildInput: 原对象或追加 memory slots 后的新对象。

    Raises:
        IrisMemoryError: 当显式查询缺少 memory service 或构建失败时抛出。
    """
    if options.memory_results is None and options.memory_query is None:
        return context_input

    if options.memory_results is not None:
        bundle = memory_context_builder.build(
            options.memory_results,
            max_chars=options.memory_max_chars,
        )
    elif options.memory_query is not None:
        if memory_service is None:
            raise IrisMemoryError("显式 memory_query 需要注入 memory_service")
        bundle = memory_service.build_context(
            options.memory_query,
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


__all__ = ["prepare_memory_context_input"]
