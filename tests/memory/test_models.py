from __future__ import annotations

import pytest
from pydantic import ValidationError

from iris.memory import (
    MemoryArtifactRef,
    MemoryQuery,
    MemoryScope,
    MemoryVisibility,
    MemoryWriteInput,
    WORKSPACE_SHARED_AGENT_ID,
    WORKSPACE_SHARED_COLLECTION,
    workspace_shared_scope,
)


def test_session_visibility_requires_session_id() -> None:
    with pytest.raises(ValidationError):
        MemoryScope(
            workspace_id="workspace",
            agent_id="agent",
            collection="default",
            visibility=MemoryVisibility.SESSION,
        )


def test_scope_text_fields_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        MemoryScope(workspace_id=" ", agent_id="agent", collection="default")


def test_workspace_shared_scope_uses_stable_convention() -> None:
    scope = workspace_shared_scope("workspace")

    assert scope.workspace_id == "workspace"
    assert scope.agent_id == WORKSPACE_SHARED_AGENT_ID
    assert scope.collection == WORKSPACE_SHARED_COLLECTION
    assert scope.visibility == MemoryVisibility.WORKSPACE
    assert scope.session_id is None


def test_query_limit_is_positive_and_capped() -> None:
    scope = MemoryScope(workspace_id="workspace", agent_id="agent", collection="default")

    query = MemoryQuery(scope=scope, text="preference", limit=1000)

    assert query.limit == 100
    with pytest.raises(ValidationError):
        MemoryQuery(scope=scope, text="preference", limit=0)


def test_write_input_rejects_blank_text_and_invalid_scores() -> None:
    scope = MemoryScope(workspace_id="workspace", agent_id="agent", collection="default")

    with pytest.raises(ValidationError):
        MemoryWriteInput(scope=scope, text=" ", reason="user asked to persist")

    with pytest.raises(ValidationError):
        MemoryWriteInput(
            scope=scope,
            text="用户偏好简洁回答",
            reason="user asked to persist",
            confidence=1.5,
        )


def test_artifact_ref_rejects_absolute_path() -> None:
    with pytest.raises(ValidationError):
        MemoryArtifactRef(path="C:/outside.txt")
