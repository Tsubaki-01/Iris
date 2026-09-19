"""Namespace 模型是新 SDK 的唯一 raw 输入契约。"""

import pytest
from pydantic import ValidationError

from iris.memory.models import (
    MemoryCandidate,
    MemoryContextFragment,
    MemoryEpisode,
    MemoryEvent,
    MemoryItem,
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
