"""Namespace 模型是新 SDK 的唯一 raw 输入契约。"""

import pytest
from pydantic import ValidationError

from iris.memory.models import (
    MemoryCandidate,
    MemoryEpisode,
    MemoryEvent,
    MemoryItem,
    MemoryItemPatch,
    MemoryObserveInput,
    MemoryOverviewConfig,
    MemoryOverviewContent,
    MemorySearchQuery,
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
        (MemorySearchQuery, {"query": "fact"}),
    ],
)
def test_models_reject_removed_scope_instead_of_defaulting_to_project(
    model: type, data: dict[str, object]
) -> None:
    with pytest.raises(ValidationError, match="scope"):
        model(**data, scope={"workspace_id": "workspace", "agent_id": "agent"})


def test_search_query_defaults_and_empty_query_are_explicit() -> None:
    """搜索的唯一输入不携带宿主作用域；空文本由搜索返回空结果。"""
    query = MemorySearchQuery(query="")
    assert query.categories == []
    assert query.kinds == []
    assert query.limit == 8
    assert MemorySearchQuery(query="fact", limit=1).limit == 1
    assert MemorySearchQuery(query="fact", limit=100).limit == 100


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"query": None},
        {"query": "fact", "limit": 0},
        {"query": "fact", "limit": 101},
        {"query": "fact", "categories": ["invalid"]},
        {"query": "fact", "kinds": ["invalid"]},
        {"query": "fact", "namespaces": ["project"]},
        {"query": "fact", "max_query_terms": 64},
        {"query": "fact", "item_ids": ["item"]},
        {"query": "fact", "include_deleted": True},
        {"query": "fact", "text": "fact"},
    ],
)
def test_search_query_rejects_invalid_or_removed_inputs(data: dict[str, object]) -> None:
    """范围与过滤字段只由唯一公共模型校验。"""
    with pytest.raises(ValidationError):
        MemorySearchQuery.model_validate(data)


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


def test_overview_config_keeps_generation_and_window_budgets_separate() -> None:
    """生成输入、生成输出和后续窗口比例各自保留明确默认值。"""
    config = MemoryOverviewConfig()
    assert config.input_budget_tokens == 96000
    assert config.max_tokens == 1024
    assert config.system_budget_ratio == 0.02
    assert MemoryOverviewConfig(input_budget_tokens=1, max_tokens=1, system_budget_ratio=1)


@pytest.mark.parametrize(
    "data",
    [
        {"input_budget_tokens": 0},
        {"max_tokens": 0},
        {"system_budget_ratio": 0},
        {"system_budget_ratio": 1.01},
        {"unknown": True},
    ],
)
def test_overview_config_rejects_invalid_budget_and_unknown_fields(
    data: dict[str, object],
) -> None:
    """预算边界由配置模型一次声明，未知配置不被静默接受。"""
    with pytest.raises(ValidationError):
        MemoryOverviewConfig.model_validate(data)


@pytest.mark.parametrize("core_facts", ["", "用户偏好中文回答。"])
def test_overview_content_accepts_required_core_facts_including_empty(core_facts: str) -> None:
    """核心事实字段必须存在，但没有核心事实时允许空字符串。"""
    content = MemoryOverviewContent(core_facts=core_facts, knowledge_scope="回答偏好与项目约定。")
    assert content.core_facts == core_facts
    assert content.knowledge_scope == "回答偏好与项目约定。"


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"knowledge_scope": "项目约定"},
        {"core_facts": ""},
        {"core_facts": "", "knowledge_scope": ""},
        {"core_facts": "", "knowledge_scope": " \n\t "},
        {"core_facts": None, "knowledge_scope": "项目约定"},
        {"core_facts": 1, "knowledge_scope": "项目约定"},
        {"core_facts": "", "knowledge_scope": ["项目约定"]},
        {"core_facts": "", "knowledge_scope": "项目约定", "version": 1},
    ],
)
def test_overview_content_rejects_missing_invalid_or_extra_fields(data: dict[str, object]) -> None:
    """双字段内容不接受缺失、空白知识范围、错误类型和额外字段。"""
    with pytest.raises(ValidationError):
        MemoryOverviewContent.model_validate(data)
