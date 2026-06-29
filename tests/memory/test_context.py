from __future__ import annotations

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    MemoryCategory,
    MemoryContextBuilder,
    MemoryItem,
    MemoryItemKind,
    MemoryLevel,
    MemoryQuery,
    MemoryScope,
    MemorySearchResult,
)


def test_context_builder_preserves_order_warns_and_counts_omitted() -> None:
    scope = MemoryScope(
        workspace_id="workspace", agent_id="agent", collection="default"
    )
    results = [
        MemorySearchResult(
            item=MemoryItem(scope=scope, text="alpha memory", reason="test seed"),
            score=2.0,
        ),
        MemorySearchResult(
            item=MemoryItem(
                scope=scope, text="beta memory that is long", reason="test seed"
            ),
            score=1.0,
        ),
    ]

    bundle = MemoryContextBuilder().build(results, max_chars=18)

    assert [fragment.item_id for fragment in bundle.fragments] == [results[0].item.id]
    assert bundle.fragments[0].warning
    assert bundle.total_chars == len("alpha memory")
    assert bundle.omitted_count == 1


def test_context_builder_copies_memory_item_semantics() -> None:
    scope = MemoryScope(
        workspace_id="workspace", agent_id="agent", collection="default"
    )
    result = MemorySearchResult(
        item=MemoryItem(
            scope=scope,
            text="用户偏好简洁回答",
            reason="用户显式说明",
            category=MemoryCategory.USER,
            kind=MemoryItemKind.PREFERENCE,
            level=MemoryLevel.SEMANTIC,
            confidence=0.8,
            importance=0.7,
        ),
        score=2.0,
        source="sqlite",
    )

    bundle = MemoryContextBuilder().build([result], max_chars=100)
    fragment = bundle.fragments[0]

    assert fragment.item_id == result.item.id
    assert fragment.text == "用户偏好简洁回答"
    assert fragment.category == MemoryCategory.USER
    assert fragment.kind == MemoryItemKind.PREFERENCE
    assert fragment.level == MemoryLevel.SEMANTIC
    assert fragment.reason == "用户显式说明"
    assert fragment.confidence == 0.8
    assert fragment.importance == 0.7
    assert "score" not in type(fragment).model_fields
    assert "source" not in type(fragment).model_fields


def test_context_builder_keeps_semantics_when_first_fragment_is_truncated() -> None:
    scope = MemoryScope(
        workspace_id="workspace", agent_id="agent", collection="default"
    )
    result = MemorySearchResult(
        item=MemoryItem(
            scope=scope,
            text="alpha memory that must be truncated",
            reason="test seed",
            category=MemoryCategory.TASK,
            kind=MemoryItemKind.TASK_STATE,
            level=MemoryLevel.EPISODIC,
            confidence=0.6,
            importance=0.9,
        ),
        score=2.0,
        source="sqlite",
    )

    bundle = MemoryContextBuilder().build([result], max_chars=5)
    fragment = bundle.fragments[0]

    assert fragment.text == "alpha"
    assert fragment.truncated is True
    assert fragment.category == MemoryCategory.TASK
    assert fragment.kind == MemoryItemKind.TASK_STATE
    assert fragment.level == MemoryLevel.EPISODIC
    assert fragment.reason == "test seed"
    assert fragment.confidence == 0.6
    assert fragment.importance == 0.9


def test_context_builder_rejects_non_positive_budget() -> None:
    with pytest.raises(IrisMemoryError):
        MemoryContextBuilder().build([], max_chars=0)


def test_memory_query_can_be_used_by_context_builder() -> None:
    scope = MemoryScope(
        workspace_id="workspace", agent_id="agent", collection="default"
    )

    query = MemoryQuery(scope=scope, text="alpha", limit=5)

    assert query.text == "alpha"
    assert query.limit == 5
