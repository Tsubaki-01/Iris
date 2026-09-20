"""索引、查询和原文片段共用完整词法。"""

import pytest

from iris.memory._query import iter_token_spans, make_snippet, prepare_fts_query, tokenize_text


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


def test_shared_lexer_reports_original_unicode_spans() -> None:
    assert list(iter_token_spans("🙂中文回答 A_b 中")) == [
        ("中文", 1, 3),
        ("文回", 2, 4),
        ("回答", 3, 5),
        ("a", 6, 7),
        ("b", 8, 9),
        ("中", 10, 11),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A B C A D", '"a" OR "b" OR "c" OR "d"'),
        ('"！？_ -', ""),
        ("OR NEAR prefix*", '"or" OR "near" OR "prefix"'),
    ],
)
def test_query_quotes_every_literal_term_once(text: str, expected: str) -> None:
    assert prepare_fts_query(tokenize_text(text)) == expected


def test_query_keeps_all_terms_without_a_budget() -> None:
    terms = [f"term{index}" for index in range(300)]
    assert prepare_fts_query(tokenize_text(" ".join(terms))) == " OR ".join(
        f'"{term}"' for term in terms
    )


def test_snippet_uses_lexical_matches_instead_of_substring_prefixes() -> None:
    text = "needlework " + "🙂" * 400 + " NEEDLE " + "🙂" * 400
    assert make_snippet(text, {"needle"}) == (text[262:562], False)
