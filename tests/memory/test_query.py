"""普通文本词法、词项预算与联合 namespace 检索。"""

from pathlib import Path

import pytest

from iris.memory._query import prepare_fts_query, tokenize_text
from iris.memory.models import MemoryEvent, MemoryEventType, MemoryItem, MemoryQuery
from iris.memory.sqlite import SQLiteMemoryStore


def test_index_tokenizer_retains_frequency_and_language_boundaries() -> None:
    assert tokenize_text("中文回答 a_B 123 A！中") == [
        "中文",
        "文回",
        "回答",
        "a",
        "b",
        "123",
        "a",
        "中",
    ]


@pytest.mark.parametrize(
    ("text", "budget", "expected"),
    [
        ("A B C A D", None, '"a" OR "b" OR "c" OR "d"'),
        ("A B C A D", 4, '"a" OR "b" OR "c" OR "d"'),
        ("A B C A D", 3, '"a" OR "d"'),
        ("A B C A D", 2, '"a" OR "d"'),
        ("A B C A D", 1, '"d"'),
        ("A B C D E", 3, '"a" OR "d" OR "e"'),
        ('"！？_ -', None, ""),
    ],
)
def test_query_budget_uses_last_occurrence_and_does_not_refill(
    text: str, budget: int | None, expected: str
) -> None:
    assert prepare_fts_query(text, max_query_terms=budget) == expected


def _add(store: SQLiteMemoryStore, text: str, namespace: str = "project") -> MemoryItem:
    item = MemoryItem(namespace=namespace, text=text)
    store.add_item(item, event=MemoryEvent(namespace=namespace, event_type=MemoryEventType.ADD))
    return item


def test_text_search_handles_chinese_punctuation_and_literal_operators(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    chinese = _add(store, "中文回答")
    punctuation = _add(store, 'literal OR "NEAR":prefix')
    assert store.search(MemoryQuery(text="请用中文回答！"))[0].item.id == chinese.id
    assert store.search(MemoryQuery(text='"NEAR":prefix OR'))[0].item.id == punctuation.id
    assert store.search(MemoryQuery(text="！？？_")) == []
    assert store.search(MemoryQuery()) == []
    assert store.search(MemoryQuery(text="unmatchedterm")) == []


def test_default_query_preserves_middle_terms_and_budget_only_changes_query(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    text = " ".join(f"term{number}" for number in range(180))
    item = _add(store, "term90")
    assert store.search(MemoryQuery(text=text))[0].item.id == item.id
    assert store.search(MemoryQuery(text=text, max_query_terms=2)) == []
    indexed = _add(store, text)
    assert indexed.id in {result.item.id for result in store.search(MemoryQuery(text="term90"))}


def test_namespaces_are_combined_before_global_rank_and_limit(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    _add(store, "needle " + "noise " * 30, "private")
    best = _add(store, "needle", "project")
    _add(store, "needle", "excluded")
    results = store.search(MemoryQuery(namespaces=["private", "project"], text="needle", limit=1))
    assert [result.item.id for result in results] == [best.id]
    assert store.get_item(best.id, ["private"]) is None
    assert store.get_item(best.id, ["private", "project"]) == best
    latest = _add(store, "latest", "project")
    assert store.list_items(["private", "project"], limit=1)[0].id == latest.id


def test_text_and_explicit_ids_are_intersected(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    item = _add(store, "needle")
    other = _add(store, "different")
    assert store.search(MemoryQuery(text="needle", item_ids=[other.id])) == []
    assert store.search(MemoryQuery(text="!", item_ids=[item.id])) == []
    assert store.search(MemoryQuery(text="needle", item_ids=[item.id]))[0].item.id == item.id
    assert store.search(MemoryQuery(item_ids=[item.id]))[0].item.id == item.id


def test_equal_rank_has_stable_order(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    for item_id in ["a", "c", "b"]:
        item = MemoryItem(id=item_id, text="needle", updated_at="2026-09-19T00:00:00")
        store.add_item(item, event=MemoryEvent(event_type=MemoryEventType.ADD))
    assert [result.item.id for result in store.search(MemoryQuery(text="needle"))] == [
        "c",
        "b",
        "a",
    ]
