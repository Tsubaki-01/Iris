"""真实 SQLite Search 的过滤、排序、完整词法和原文摘要合同。"""

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from iris.memory import (
    MemoryArtifactRef,
    MemoryCandidate,
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryItemKind,
    MemoryObserveInput,
    MemorySearchHit,
    MemorySearchQuery,
    MemorySearchResponse,
    MemoryService,
    SQLiteMemoryStore,
)


def _add(store: SQLiteMemoryStore, text: str, **values: Any) -> MemoryItem:
    item = MemoryItem(text=text, **values)
    return store.add_item(
        item, event=MemoryEvent(namespace=item.namespace, event_type=MemoryEventType.ADD)
    )


def test_search_returns_only_the_six_hit_fields(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "search.db")
    item = _add(store, "Needle 原文。", metadata={"private": "detail"})
    response = store.search(MemorySearchQuery(query="needle"), ["project"])
    assert response == MemorySearchResponse(
        items=(MemorySearchHit(item.id, "project", item.category, item.kind, item.text, True),),
        has_more=False,
    )
    assert {field.name for field in fields(response.items[0])} == {
        "item_id",
        "namespace",
        "category",
        "kind",
        "snippet",
        "is_complete",
    }


@pytest.mark.parametrize("query", ["", " \n\t ", '"！？_ -', "🙂 русский", "missing"])
def test_search_with_no_terms_or_hits_returns_an_empty_response(tmp_path: Path, query: str) -> None:
    store = SQLiteMemoryStore(tmp_path / "empty.db")
    _add(store, "stored fact")
    assert store.search(MemorySearchQuery(query=query), ["project"]) == MemorySearchResponse(
        (), False
    )


def test_no_allowed_namespace_returns_no_memory(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "scope.db")
    _add(store, "needle")
    assert store.search(MemorySearchQuery(query="needle"), []) == MemorySearchResponse((), False)


def test_required_terms_filter_each_result_before_limit_without_relaxing(tmp_path: Path) -> None:
    """一个正文满足主题 OR 与全部必要词；已有范围、状态和 limit 仍生效。"""
    store = SQLiteMemoryStore(tmp_path / "required.db")
    first = _add(store, "alpha east prod")
    second = _add(store, "beta east prod")
    _add(store, "alpha east test")
    _add(store, "beta west prod")
    _add(store, "gamma east prod")
    _add(store, "alpha east prod", namespace="private")
    _add(store, "alpha east prod", status="deleted")
    _add(store, "alpha west", id="east_prod", metadata={"environment": "east prod"})
    query = MemorySearchQuery(query="alpha beta", required_terms=["east", "prod"], limit=1)
    response = store.search(query, ["project"])
    assert len(response.items) == 1 and response.has_more
    assert response.items[0].item_id in {first.id, second.id}
    response = store.search(query.model_copy(update={"limit": 8}), ["project"])
    assert {hit.item_id for hit in response.items} == {first.id, second.id}
    assert not response.has_more
    for impossible in (
        MemorySearchQuery(query="alpha", required_terms=["absent"]),
        MemorySearchQuery(query="!!!", required_terms=["east"]),
    ):
        assert store.search(impossible, ["project"]) == MemorySearchResponse((), False)


@pytest.mark.parametrize(
    ("phrase", "matching", "nonmatching"),
    [
        ("RED, BLUE", "red blue", "red green blue"),
        ("red blue", "red-blue", "blue red"),
        ("go go", "go go now", "go now"),
        ("人人人", "人人人", "人人"),
        ("账单导出", "账单导出", "账单-导出"),
        ("中", "中", "中文"),
    ],
)
def test_required_phrase_uses_index_order_and_language_boundaries(
    tmp_path: Path, phrase: str, matching: str, nonmatching: str
) -> None:
    """硬词组匹配的是索引词序列，不是无序词集合或任意子串。"""
    store = SQLiteMemoryStore(tmp_path / "phrase.db")
    expected = _add(store, f"needle {matching}")
    _add(store, f"needle {nonmatching}")
    response = store.search(MemorySearchQuery(query="needle", required_terms=[phrase]), ["project"])
    assert [hit.item_id for hit in response.items] == [expected.id]


def test_required_phrase_has_no_term_cap_and_does_not_move_the_snippet(tmp_path: Path) -> None:
    """必要词组完整进入 FTS；摘要仍围绕普通 query，长条目可继续 Fetch。"""
    store = SQLiteMemoryStore(tmp_path / "long-phrase.db")
    phrase = " ".join(f"term{index}" for index in range(150))
    body = "needle " + "🙂" * 400 + " " + phrase
    expected = _add(store, body)
    _add(store, "needle " + " ".join(f"term{index}" for index in range(149)))
    response = store.search(MemorySearchQuery(query="needle", required_terms=[phrase]), ["project"])
    assert [hit.item_id for hit in response.items] == [expected.id]
    assert response.items[0].snippet == body[:300]
    assert not response.items[0].is_complete


def test_filtering_precedes_rank_and_limit_with_or_within_each_dimension(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "filters.db")
    _add(store, "needle", namespace="excluded", category="user", kind="fact")
    _add(store, "needle", namespace="project", category="task", kind="fact")
    _add(store, "needle", namespace="project", category="user", kind="correction")
    first = _add(store, "needle " + "noise " * 12, namespace="project", kind="fact")
    second = _add(store, "needle " + "noise " * 30, namespace="private", category="reference")
    query = MemorySearchQuery(
        query="needle",
        required_terms=["needle"],
        categories=["user", "reference"],
        kinds=["fact", "note"],
        limit=1,
    )
    response = store.search(query, ["private", "project"])
    assert [hit.item_id for hit in response.items] == [first.id]
    assert response.has_more
    response = store.search(query.model_copy(update={"limit": 2}), ["project", "private"])
    assert [hit.item_id for hit in response.items] == [first.id, second.id]
    assert not response.has_more


def test_search_keeps_active_l1_and_l2_but_excludes_other_statuses(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "active.db")
    l1 = _add(store, "needle", level="l1")
    l2 = _add(store, "needle", level="l2")
    _add(store, "needle", status="deleted")
    _add(store, "needle", status="superseded")
    response = store.search(MemorySearchQuery(query="needle"), ["project"])
    assert {hit.item_id for hit in response.items} == {l1.id, l2.id}


def test_only_item_text_is_searchable(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "text-only.db")
    _add(
        store,
        "ordinary body",
        id="needle",
        namespace="needle",
        source_id="needle",
        reason="needle",
        metadata={"needle": "needle"},
        artifacts=[MemoryArtifactRef(path="needle.txt", metadata={"needle": "needle"})],
    )
    assert store.search(MemorySearchQuery(query="needle"), ["needle"]).items == ()


def test_episode_and_candidate_are_not_searchable_until_promotion(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "promotion.db")
    service = MemoryService(store)
    episode = service.observe(MemoryObserveInput(text="needle episode"))
    candidate = service.add_candidate(
        MemoryCandidate(episode_ids=[episode.id], text="needle candidate", reason="extract")
    )
    query = MemorySearchQuery(query="needle")
    assert service.search(query, ["project"]).items == ()
    promoted = service.promote_candidate(
        candidate.id, "project", kind=MemoryItemKind.FACT, reason="reviewed"
    )
    assert [hit.item_id for hit in service.search(query, ["project"]).items] == [promoted.id]


def test_query_and_index_keep_terms_beyond_128_and_far_from_both_ends(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "all-terms.db")
    query = " ".join(f"term{number}" for number in range(300))
    item = _add(store, "term150")
    assert store.search(MemorySearchQuery(query=query), ["project"]).items[0].item_id == item.id
    indexed = _add(store, query)
    assert indexed.id in {
        hit.item_id for hit in store.search(MemorySearchQuery(query="term150"), ["project"]).items
    }


def test_literal_operators_and_chinese_use_the_index_lexer(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "literal.db")
    chinese = _add(store, "中文回答")
    literal = _add(store, 'literal OR "NEAR":prefix')
    assert (
        store.search(MemorySearchQuery(query="请用中文回答！"), ["project"]).items[0].item_id
        == chinese.id
    )
    assert (
        store.search(MemorySearchQuery(query='"NEAR":prefix OR'), ["project"]).items[0].item_id
        == literal.id
    )
    assert store.search(MemorySearchQuery(query="中"), ["project"]).items == ()


def test_same_text_different_ids_remain_and_ties_sort_by_updated_at_then_id(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "ties.db")
    for item_id, date in [("a", "2026-09-20"), ("c", "2026-09-19"), ("b", "2026-09-20")]:
        _add(store, "needle needle", id=item_id, updated_at=date)
    response = store.search(
        MemorySearchQuery(query="needle needle", limit=2), ["project", "project"]
    )
    assert [hit.item_id for hit in response.items] == ["b", "a"]
    assert response.has_more


@pytest.mark.parametrize(
    ("text", "query", "expected_start"),
    [
        ("needle " + "🙂" * 293, "needle", None),
        ("needle " + "🙂" * 294, "needle", 0),
        ("🙂" * 600 + " NEEDLE tail", "needle", 312),
        ("🙂" * 400 + " first " + "🙂" * 400 + " second", "second first", 251),
        ("needlework " + "🙂" * 400 + " NEEDLE " + "🙂" * 400, "needle", 262),
        ("🙂" * 400 + " 中文回答 " + "🙂" * 400, "中文回答", 251),
        ("🙂" * 200 + " " + "A" * 400, "A" * 400, 51),
    ],
)
def test_snippet_is_the_exact_unicode_window_around_the_first_matching_token(
    tmp_path: Path, text: str, query: str, expected_start: int | None
) -> None:
    store = SQLiteMemoryStore(tmp_path / "snippet.db")
    _add(store, text)
    hit = store.search(MemorySearchQuery(query=query), ["project"]).items[0]
    if expected_start is None:
        assert hit.snippet == text
        assert hit.is_complete
    else:
        assert hit.snippet == text[expected_start : expected_start + 300]
        assert len(hit.snippet) == 300
        assert not hit.is_complete
