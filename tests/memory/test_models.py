"""Namespace 模型是新 SDK 的唯一 raw 输入契约。"""

import pytest
from pydantic import ValidationError

from iris.memory.models import (
    MemoryCandidate,
    MemoryContextFragment,
    MemoryEpisode,
    MemoryEvent,
    MemoryItem,
    MemoryItemPatch,
    MemoryObserveInput,
    MemoryQuery,
    MemoryWriteInput,
)


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (MemoryItem, {"text": "item"}),
        (MemoryEpisode, {}),
        (MemoryEvent, {"event_type": "add"}),
        (MemoryCandidate, {"text": "candidate", "reason": "reason", "episode_ids": ["e"]}),
        (MemoryObserveInput, {}),
        (MemoryWriteInput, {"text": "item", "reason": "reason"}),
        (MemoryQuery, {}),
    ],
)
def test_models_reject_removed_scope_instead_of_defaulting_to_project(
    model: type, data: dict[str, object]
) -> None:
    with pytest.raises(ValidationError, match="scope"):
        model(**data, scope={"workspace_id": "workspace", "agent_id": "agent"})


def test_query_defaults_and_budget_boundary() -> None:
    query = MemoryQuery()
    assert query.namespaces == ["project"]
    assert query.max_query_terms is None
    for changes in [{"namespaces": []}, {"namespaces": [" "]}, {"max_query_terms": 0}]:
        with pytest.raises(ValidationError):
            MemoryQuery(**changes)
    assert MemoryQuery(max_query_terms=1).max_query_terms == 1


def test_context_fragment_carries_namespace() -> None:
    fragment = MemoryContextFragment(
        item_id="i",
        namespace="project",
        text="text",
        category="user",
        kind="note",
        level="l2",
        warning="warning",
    )
    assert fragment.namespace == "project"


@pytest.mark.parametrize("field", ["text", "category", "kind", "status", "artifacts", "metadata"])
def test_patch_rejects_explicit_null_for_non_nullable_item_fields(field: str) -> None:
    """字段可以省略，但不能将 MemoryItem 的必需字段显式改为 null。"""
    with pytest.raises(ValidationError):
        MemoryItemPatch.model_validate({field: None})


def test_patch_preserves_omission_and_allows_nullable_scores_to_clear() -> None:
    """省略、清空评分和清空集合是三种合法且不同的更新。"""
    assert MemoryItemPatch().model_dump(exclude_unset=True) == {}
    assert MemoryItemPatch(confidence=None, importance=None).model_dump(exclude_unset=True) == {
        "confidence": None,
        "importance": None,
    }
    assert MemoryItemPatch(artifacts=[], metadata={}).model_dump(exclude_unset=True) == {
        "artifacts": [],
        "metadata": {},
    }
