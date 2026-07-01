from __future__ import annotations

from pathlib import Path

from iris.memory import (
    MemoryActor,
    MemoryCandidateStatus,
    MemoryCategory,
    MemoryEventType,
    MemoryItem,
    MemoryItemKind,
    MemoryObserveInput,
    MemoryOrchestrator,
    MemoryQuery,
    MemoryScope,
    MemoryService,
    MemorySourceType,
    RuleMemoryExtractor,
    SQLiteMemoryStore,
)


def test_orchestrator_noop_by_default_creates_no_candidates(tmp_path: Path) -> None:
    service = _service(tmp_path)
    scope = _scope()
    orchestrator = MemoryOrchestrator(service)

    candidates = orchestrator.observe(
        MemoryObserveInput(
            scope=scope,
            text="用户希望回答简短",
            source_type=MemorySourceType.MESSAGE,
            source_id="msg_1",
        )
    )

    assert candidates == []
    assert service.list_candidates(scope) == []
    assert service.list_items(scope) == []
    assert [event.event_type for event in service.list_events(scope)] == [
        MemoryEventType.OBSERVE
    ]


def test_rule_extractor_creates_pending_candidate_without_l2_promotion(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    scope = _scope()
    orchestrator = MemoryOrchestrator(service, extractor=RuleMemoryExtractor())

    candidates = orchestrator.observe(
        MemoryObserveInput(
            scope=scope,
            text="用户偏好简洁中文回答",
            category=MemoryCategory.USER,
            metadata={"memory_kind": "preference"},
        )
    )

    assert len(candidates) == 1
    assert candidates[0].status == MemoryCandidateStatus.PENDING
    assert service.list_items(scope) == []
    assert service.list_candidates(scope, status=MemoryCandidateStatus.PENDING)[
        0
    ].id == (candidates[0].id)


def test_process_candidates_promotes_only_on_explicit_call(tmp_path: Path) -> None:
    service = _spy_service(tmp_path)
    scope = _scope()
    orchestrator = MemoryOrchestrator(service, extractor=RuleMemoryExtractor())
    candidate = orchestrator.observe(
        MemoryObserveInput(
            scope=scope,
            text="用户偏好简洁中文回答",
            category=MemoryCategory.USER,
            metadata={"memory_kind": "preference"},
        )
    )[0]

    assert service.recall(MemoryQuery(scope=scope, text="简洁")) == []

    items = orchestrator.process_candidates(scope)

    assert [item.text for item in items] == ["用户偏好简洁中文回答"]
    assert items[0].episode_id == candidate.episode_ids[0]
    assert items[0].kind.value == "preference"
    assert service.promoted_candidate_ids == [candidate.id]
    assert service.list_candidates(scope)[0].status == MemoryCandidateStatus.ACCEPTED
    assert (
        service.recall(MemoryQuery(scope=scope, text="简洁"))[0].item.id == items[0].id
    )


def test_low_confidence_candidate_is_rejected_and_audited(tmp_path: Path) -> None:
    service = _service(tmp_path)
    scope = _scope()
    orchestrator = MemoryOrchestrator(service, extractor=RuleMemoryExtractor())

    candidates = orchestrator.observe(
        MemoryObserveInput(
            scope=scope,
            text="低置信候选",
            category=MemoryCategory.USER,
            metadata={"memory_confidence": 0.2, "memory_importance": 0.8},
        )
    )

    assert candidates[0].status == MemoryCandidateStatus.REJECTED
    assert orchestrator.process_candidates(scope) == []
    assert service.list_items(scope) == []
    events = service.list_events(scope)
    assert MemoryEventType.CANDIDATE_ADD in {event.event_type for event in events}
    assert candidates[0].id in {
        event.metadata.get("candidate_id") for event in events if event.metadata
    }


def _service(tmp_path: Path) -> MemoryService:
    return MemoryService(SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False))


def _spy_service(tmp_path: Path) -> _SpyMemoryService:
    return _SpyMemoryService(SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False))


class _SpyMemoryService(MemoryService):
    def __init__(self, store: SQLiteMemoryStore) -> None:
        super().__init__(store)
        self.promoted_candidate_ids: list[str] = []

    def promote_candidate(
        self,
        candidate_id: str,
        scope: MemoryScope,
        *,
        kind: MemoryItemKind,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> MemoryItem:
        self.promoted_candidate_ids.append(candidate_id)
        return super().promote_candidate(
            candidate_id,
            scope,
            kind=kind,
            actor=actor,
            reason=reason,
        )


def _scope(*, agent_id: str = "agent") -> MemoryScope:
    return MemoryScope(
        workspace_id="workspace", agent_id=agent_id, collection="default"
    )
