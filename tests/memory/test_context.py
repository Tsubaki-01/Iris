from __future__ import annotations

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    MemoryContextBuilder,
    MemoryItem,
    MemoryQuery,
    MemoryScope,
    MemorySearchResult,
)


def test_context_builder_preserves_order_warns_and_counts_omitted() -> None:
    scope = MemoryScope(workspace_id="workspace", agent_id="agent", collection="default")
    results = [
        MemorySearchResult(
            item=MemoryItem(scope=scope, text="alpha memory", reason="test seed"),
            score=2.0,
        ),
        MemorySearchResult(
            item=MemoryItem(scope=scope, text="beta memory that is long", reason="test seed"),
            score=1.0,
        ),
    ]

    bundle = MemoryContextBuilder().build(results, max_chars=18)

    assert [fragment.item_id for fragment in bundle.fragments] == [results[0].item.id]
    assert bundle.fragments[0].warning
    assert bundle.total_chars == len("alpha memory")
    assert bundle.omitted_count == 1


def test_context_builder_rejects_non_positive_budget() -> None:
    with pytest.raises(IrisMemoryError):
        MemoryContextBuilder().build([], max_chars=0)


def test_memory_query_can_be_used_by_context_builder() -> None:
    scope = MemoryScope(workspace_id="workspace", agent_id="agent", collection="default")

    query = MemoryQuery(scope=scope, text="alpha", limit=5)

    assert query.text == "alpha"
    assert query.limit == 5
