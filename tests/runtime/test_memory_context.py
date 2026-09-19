"""动态片段按来源与实际渲染内容去重，不扩展到摘要或语义相似度。"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris.context import ContextBuilder
from iris.exceptions import IrisContextError
from iris.lifecycle import RuntimeExecutionOptions
from iris.memory import (
    MemoryConfig,
    MemoryContextBuilder,
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
    MemoryService,
    SQLiteMemoryStore,
)
from iris.message import Msg
from iris.runtime.memory_context import prepare_run_memory_messages


@pytest.mark.asyncio
async def test_only_auto_recall_deduplicates_visible_identical_fragments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ID、正文、截断结果决定是否重复；排名变化不应制造新快照。"""
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"))
    result = MemorySearchResult(item=MemoryItem(id="same-id", text="abcdef"))
    results = [result]
    calls: list[MemoryQuery] = []

    async def recall(query: MemoryQuery) -> list[MemorySearchResult]:
        calls.append(query)
        return results

    monkeypatch.setattr(service, "arecall", recall)

    async def prepare(
        visible: list[Msg], *, options: RuntimeExecutionOptions | None = None
    ) -> tuple[Msg, ...]:
        return await prepare_run_memory_messages(
            options=options or RuntimeExecutionOptions(),
            memory_service=service,
            memory_context_builder=MemoryContextBuilder(),
            context_builder=ContextBuilder(),
            config=MemoryConfig(),
            run_input="abc",
            run_id="dedup",
            visible_history=visible,
        )

    original = list(await prepare([]))
    assert len(original) == 1
    assert await prepare(original) == ()
    results = [result.model_copy(update={"score": 99.0, "matched_text": "different query"})]
    assert await prepare(original) == ()
    # 正文、ID或实际截断变化，均不视为已经完整看过的同一片段。
    results = [
        result.model_copy(update={"item": result.item.model_copy(update={"text": "abcdEF"})})
    ]
    assert len(await prepare(original)) == 1
    results = [
        result.model_copy(update={"item": result.item.model_copy(update={"id": "other-id"})})
    ]
    assert len(await prepare(original)) == 1
    results = [result]
    truncated = list(await prepare([], options=RuntimeExecutionOptions(memory_max_chars=3)))
    assert len(await prepare(truncated)) == 1
    assert len(await prepare(original, options=RuntimeExecutionOptions(memory_max_chars=3))) == 1

    # 普通工具输出、摘要和静态内容不具备动态快照的来源标记。
    for visible in (
        [Msg.user(original[0].text, sender="context")],
        [Msg.user("<summary>abcdef</summary>", sender="context")],
        [Msg.tool_result(tool_use_id="read", content=original[0].text)],
    ):
        assert len(await prepare(visible)) == 1

    explicit = RuntimeExecutionOptions(memory_results=[result.model_dump(mode="json")])
    prior_calls = len(calls)
    repeated = await prepare(original, options=explicit)
    assert len(repeated) == 1
    assert repeated[0].text == original[0].text
    assert repeated[0].metadata == original[0].metadata
    assert len(calls) == prior_calls
    # 显式动态快照也能为之后自动召回提供原文证据。
    assert await prepare(list(await prepare([], options=explicit))) == ()


@pytest.mark.asyncio
async def test_dedup_does_not_refill_after_body_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """先形成正文预算内片段，再去重；不把省出的预算重新填满。"""
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"))
    results = [MemorySearchResult(item=MemoryItem(id=key, text=key * 3)) for key in ("a", "b")]
    calls = 0

    async def recall(query: MemoryQuery) -> list[MemorySearchResult]:
        nonlocal calls
        calls += 1
        return results

    monkeypatch.setattr(service, "arecall", recall)
    first = await prepare_run_memory_messages(
        options=RuntimeExecutionOptions(memory_results=[results[0].model_dump(mode="json")]),
        memory_service=service,
        memory_context_builder=MemoryContextBuilder(),
        context_builder=ContextBuilder(),
        config=MemoryConfig(),
        run_input="a b",
        run_id="first",
        visible_history=[],
    )
    second = await prepare_run_memory_messages(
        options=RuntimeExecutionOptions(memory_max_chars=3),
        memory_service=service,
        memory_context_builder=MemoryContextBuilder(),
        context_builder=ContextBuilder(),
        config=MemoryConfig(),
        run_input="a b",
        run_id="second",
        visible_history=list(first),
    )
    assert second == ()
    assert calls == 1


@pytest.mark.asyncio
async def test_auto_recall_does_not_swallow_render_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """容错只包住自动读取，不掩盖后续 context 渲染错误。"""
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"))

    async def recall(query: MemoryQuery) -> list[MemorySearchResult]:
        return [MemorySearchResult(item=MemoryItem(text="配置资料"))]

    monkeypatch.setattr(service, "arecall", recall)
    builder = ContextBuilder()

    def fail(*args: object, **kwargs: object) -> str:
        raise IrisContextError("模板错误")

    monkeypatch.setattr(builder, "render_section", fail)
    with pytest.raises(IrisContextError, match="模板错误"):
        await prepare_run_memory_messages(
            options=RuntimeExecutionOptions(),
            memory_service=service,
            memory_context_builder=MemoryContextBuilder(),
            context_builder=builder,
            config=MemoryConfig(),
            run_input="配置",
            run_id="render-failed",
            visible_history=[],
        )
