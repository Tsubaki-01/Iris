"""Memory 的单次语义召回与共享 Search 输出契约。"""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from iris.decision import DecisionRequest, DecisionResponse, DecisionUsage, ScoreAnswer
from iris.exceptions import IrisDecisionError, IrisMemoryError
from iris.memory import (
    MEMORY_TOOL_CLASSES,
    MemoryAccessPolicy,
    MemoryCategory,
    MemoryItem,
    MemoryItemKind,
    MemorySearchQuery,
    MemorySearchTool,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
    register_memory_tools,
)
from iris.memory.recall import recall_memories
from iris.prompts import PromptSnapshot, PromptSource
from iris.tools import ToolCapability, ToolExecutionContext


@pytest.fixture
def prompt_snapshot(tmp_path: Path) -> PromptSnapshot:
    """在工具构造边界固定项目提示。"""
    return PromptSource.initialize(tmp_path).snapshot()


class FakeDecision:
    """只提供 evaluate，保留请求并按反序题号返回分数。"""

    def __init__(self, scores: tuple[float, ...], error: IrisDecisionError | None = None) -> None:
        self.scores = scores
        self.error = error
        self.requests: list[DecisionRequest] = []

    async def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return DecisionResponse(
            provider="typesafe",
            model="actual-memory-model",
            answers={
                f"m{i}": ScoreAnswer(
                    score=score,
                    probabilities={0: 0.0, 1: 0.0, 2: 0.5, 3: 0.5},
                    levels=request.questions[f"m{i}"].levels,
                    confidence=0.5,
                )
                for i, score in reversed(list(enumerate(self.scores)))
            },
            usage=DecisionUsage(input_tokens=84, output_tokens=20),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limit", "expected_ids", "has_more"),
    [(2, ["item3", "item4"], True), (5, ["item3", "item4", "item2"], False)],
)
async def test_score_boundary_stable_order_and_limit_only_affect_final_hits(
    limit: int, expected_ids: list[str], has_more: bool, prompt_snapshot: PromptSnapshot
) -> None:
    candidates = [MemoryItem(id=f"item{i}", text=f"正文{i}") for i in range(5)]
    evaluator = FakeDecision((0, 1.99, 2.0, 2.8, 2.8))
    response, metadata = await recall_memories(
        candidates, MemorySearchQuery(query="查询", limit=limit), evaluator, prompt_snapshot
    )
    assert [hit.item_id for hit in response.items] == expected_ids
    assert response.has_more is has_more
    assert len(evaluator.requests) == 1
    assert len(evaluator.requests[0].questions) == 5
    assert metadata == {
        "decision": {
            "feature": "memory.recall",
            "provider": "typesafe",
            "model": "actual-memory-model",
            "question_count": 5,
            "input_tokens": 84,
            "output_tokens": 20,
        }
    }


@pytest.mark.asyncio
async def test_recall_sends_only_query_and_filtered_bodies_and_returns_full_original_text(
    prompt_snapshot: PromptSnapshot,
) -> None:
    body = "Release checklist go go " + "🟢" * 420
    candidate = MemoryItem(
        id="local-id",
        namespace="project",
        text=body,
        category=MemoryCategory.REFERENCE,
        kind=MemoryItemKind.FACT,
        metadata={"local": "not sent"},
    )
    excluded = MemoryItem(text="Release checklist go once", metadata={"text": "go go"})
    evaluator = FakeDecision((2.4,))
    query = MemorySearchQuery(
        query="如何撤回发布",
        required_terms=["GO, go", "release checklist"],
        categories=[MemoryCategory.REFERENCE],
        kinds=[MemoryItemKind.FACT],
        limit=1,
    )
    response, _ = await recall_memories([candidate, excluded], query, evaluator, prompt_snapshot)
    request = evaluator.requests[0]
    assert request.state == {"query": "如何撤回发布", "memories": [body]}
    assert list(request.questions) == ["m0"]
    question = request.questions["m0"]
    assert question.type == "score" and len(question.levels) == 4
    assert "memories[0]" in question.instructions and "query" in question.instructions
    assert body not in question.instructions
    assert response.items[0].item_id == "local-id"
    assert response.items[0].namespace == candidate.namespace
    assert response.items[0].category == candidate.category
    assert response.items[0].kind == candidate.kind
    assert response.items[0].snippet == body
    assert response.items[0].is_complete


@pytest.mark.asyncio
@pytest.mark.parametrize("candidates", [[], [MemoryItem(text="unmatched")]])
async def test_empty_or_phrase_filtered_catalog_skips_decision(
    candidates: list[MemoryItem],
    prompt_snapshot: PromptSnapshot,
) -> None:
    evaluator = FakeDecision(())
    response, metadata = await recall_memories(
        candidates,
        MemorySearchQuery(query="question", required_terms=["required"]),
        evaluator,
        prompt_snapshot,
    )
    assert response.items == () and not response.has_more
    assert metadata == {}
    assert evaluator.requests == []


@pytest.mark.asyncio
async def test_single_phrase_matching_item_still_needs_semantic_score(
    prompt_snapshot: PromptSnapshot,
) -> None:
    evaluator = FakeDecision((0.0,))
    response, metadata = await recall_memories(
        [MemoryItem(text="go go unrelated")],
        MemorySearchQuery(query="!!!", required_terms=["go go"]),
        evaluator,
        prompt_snapshot,
    )
    assert response.items == () and not response.has_more
    assert len(evaluator.requests) == 1
    assert metadata["decision"]["question_count"] == 1


@pytest.mark.asyncio
async def test_search_reads_the_full_current_scope_once_without_lexical_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt_snapshot: PromptSnapshot
) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "recall.db"))
    selected = service.remember(
        MemoryWriteInput(
            namespace="team",
            text="release go go",
            reason="seed",
            category=MemoryCategory.REFERENCE,
            kind=MemoryItemKind.FACT,
        )
    )
    service.remember(
        MemoryWriteInput(
            namespace="project",
            text="release go now",
            reason="seed",
            category=MemoryCategory.REFERENCE,
            kind=MemoryItemKind.FACT,
        )
    )
    original_list = service.alist_items
    original_io = service.run_async_io
    calls: list[tuple[Sequence[str], dict[str, Any]]] = []
    io_calls = 0

    async def list_items(namespaces: Sequence[str], **kwargs: Any) -> list[MemoryItem]:
        calls.append((namespaces, kwargs))
        return await original_list(namespaces, **kwargs)

    async def record_io(operation: Any, **kwargs: Any) -> Any:
        nonlocal io_calls
        io_calls += 1
        return await original_io(operation, **kwargs)

    def no_search(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Decision 召回不允许 query 词法预筛")

    monkeypatch.setattr(service, "alist_items", list_items)
    monkeypatch.setattr(service, "run_async_io", record_io)
    monkeypatch.setattr(service, "search", no_search)
    monkeypatch.setattr(service, "asearch", no_search)
    evaluator = FakeDecision((2.2,))
    tool = MemorySearchTool(
        service=service,
        access_policy_factory=lambda _: MemoryAccessPolicy(read_namespaces=("team", "project")),
        decision_client=evaluator,
        prompt_snapshot=prompt_snapshot,
    )
    result = await tool.arun(
        MemorySearchQuery(
            query="如何撤回发布",
            required_terms=["go go"],
            categories=[MemoryCategory.REFERENCE],
            kinds=[MemoryItemKind.FACT],
            limit=1,
        ),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert calls == [
        (
            ["team", "project"],
            {
                "limit": None,
                "categories": [MemoryCategory.REFERENCE],
                "kinds": [MemoryItemKind.FACT],
            },
        )
    ]
    assert io_calls == 1
    assert evaluator.requests[0].state == {"query": "如何撤回发布", "memories": [selected.text]}
    payload = json.loads(result.model_content)
    assert payload == {
        "items": [
            {
                "item_id": selected.id,
                "namespace": "team",
                "category": "reference",
                "kind": "fact",
                "snippet": selected.text,
                "is_complete": True,
            }
        ],
        "has_more": False,
    }
    assert result.metadata["decision"]["question_count"] == 1


def test_shared_service_tools_have_identical_public_input_without_capability_pollution(
    tmp_path: Path,
    prompt_snapshot: PromptSnapshot,
) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "shared.db"))

    def policy(_: ToolExecutionContext) -> MemoryAccessPolicy:
        return MemoryAccessPolicy()

    local = MemorySearchTool(service=service, access_policy_factory=policy)
    remote_registry = register_memory_tools(
        service=service,
        access_policy_factory=policy,
        tool_names=tuple(MEMORY_TOOL_CLASSES),
        memory_decision_client=FakeDecision(()),
        prompt_snapshot=prompt_snapshot,
    )
    remote = remote_registry.get("memory_search")
    assert local.input_model is remote.input_model is MemorySearchQuery
    assert local.definition.input_schema == remote.definition.input_schema
    assert local.definition.description == remote.definition.description
    assert "OR" not in remote.definition.description
    assert "Jev" not in remote.definition.description
    schema = remote.definition.input_schema
    assert set(schema["properties"]) == {"query", "required_terms", "categories", "kinds", "limit"}
    assert schema["required"] == ["query"] and schema["additionalProperties"] is False
    for tool in (local, remote):
        assert tool.validate_input({"query": " ask "}).query == "ask"
        for invalid in (
            {"query": " "},
            {"query": "ask", "required_terms": ["---"]},
            {"query": "ask", "new": 1},
        ):
            with pytest.raises(ValidationError):
                tool.validate_input(invalid)
    assert local.definition.capabilities == {ToolCapability.READ}
    assert remote.definition.capabilities == {ToolCapability.READ, ToolCapability.NETWORK}
    assert remote_registry.get("memory_fetch").definition.capabilities == {ToolCapability.READ}
    for name in ("memory_remember", "memory_update", "memory_forget"):
        assert remote_registry.get(name).definition.capabilities == {ToolCapability.WRITE}
    assert local.is_read_only({}) and not remote.is_read_only({})


@pytest.mark.asyncio
@pytest.mark.parametrize(("reason", "count"), [("network failed", 1), ("capacity exceeded", 130)])
async def test_failure_preserves_all_candidates_and_never_returns_lexical_fallback(
    reason: str, count: int, prompt_snapshot: PromptSnapshot
) -> None:
    error = IrisDecisionError(reason)
    evaluator = FakeDecision((), error)
    with pytest.raises(IrisMemoryError) as caught:
        await recall_memories(
            [MemoryItem(text=f"body {i}") for i in range(count)],
            MemorySearchQuery(query="anything", limit=1),
            evaluator,
            prompt_snapshot,
        )
    assert caught.value.__cause__ is error
    assert len(evaluator.requests) == 1 and len(evaluator.requests[0].questions) == count


@pytest.mark.asyncio
async def test_outer_cancellation_propagates_to_decision(prompt_snapshot: PromptSnapshot) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingDecision:
        async def evaluate(self, request: DecisionRequest) -> DecisionResponse:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            raise AssertionError("等待必须被取消")

    task = asyncio.create_task(
        recall_memories(
            [MemoryItem(text="body")],
            MemorySearchQuery(query="anything"),
            BlockingDecision(),
            prompt_snapshot,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_existing_tool_keeps_instruction_but_reads_new_candidates(tmp_path: Path) -> None:
    source = PromptSource.initialize(tmp_path)
    template = source.root / "memory_recall_instruction.j2"
    template.write_text("old {{ candidate_index }}", encoding="utf-8")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "recall.db"))
    first = service.remember(MemoryWriteInput(text="first", reason="seed"))
    evaluator = FakeDecision((2.0,))
    old_tool = MemorySearchTool(
        service=service,
        access_policy_factory=lambda _: MemoryAccessPolicy(),
        decision_client=evaluator,
        prompt_snapshot=source.snapshot(),
    )
    context = ToolExecutionContext(workspace_root=tmp_path)
    query = MemorySearchQuery(query="query")
    await old_tool.arun(query, context)
    template.write_text("new {{ candidate_index }}", encoding="utf-8")
    second = service.remember(MemoryWriteInput(text="second", reason="seed"))
    evaluator.scores = (2.0, 2.0)
    await old_tool.arun(query, context)
    new_tool = MemorySearchTool(
        service=service,
        access_policy_factory=lambda _: MemoryAccessPolicy(),
        decision_client=evaluator,
        prompt_snapshot=source.snapshot(),
    )
    await new_tool.arun(query, context)
    assert evaluator.requests[0].state["memories"] == [first.text]
    assert set(evaluator.requests[1].state["memories"]) == {first.text, second.text}
    assert [q.instructions for q in evaluator.requests[1].questions.values()] == ["old 0", "old 1"]
    assert [q.instructions for q in evaluator.requests[2].questions.values()] == ["new 0", "new 1"]
