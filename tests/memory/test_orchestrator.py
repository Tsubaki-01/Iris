from __future__ import annotations

from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    FileMemoryMirror,
    MemoryActor,
    MemoryCandidateStatus,
    MemoryCategory,
    MemoryEventType,
    MemoryItem,
    MemoryItemKind,
    MemoryObserveInput,
    MemoryOrchestrator,
    MemoryService,
    MemoryStore,
    RuleMemoryExtractor,
    SQLiteMemoryStore,
)


@pytest.mark.parametrize("fail_third", [False, True])
def test_candidate_batch_rebuilds_once_with_all_committed_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_third: bool,
) -> None:
    namespace = "project"
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(store, mirror=mirror)
    orchestrator = MemoryOrchestrator(service, extractor=RuleMemoryExtractor())
    for index in range(3):
        orchestrator.observe(
            MemoryObserveInput(
                namespace=namespace, text=f"preference {index}", category=MemoryCategory.USER
            )
        )
    target = mirror.root / "User/user.md"
    target.write_text("manual note\n", encoding="utf-8")
    rebuild_item_counts: list[int] = []
    original_rebuild = mirror.rebuild_from_store
    original_promote = store.promote_candidate
    promotion_calls = 0

    def rebuild(current_store: MemoryStore, current_namespace: str) -> None:
        rebuild_item_counts.append(len(current_store.list_items([current_namespace])))
        original_rebuild(current_store, current_namespace)

    def promote(
        candidate_id: str,
        current_namespace: str,
        *,
        kind: MemoryItemKind,
        actor: MemoryActor,
        reason: str,
    ) -> MemoryItem:
        nonlocal promotion_calls
        promotion_calls += 1
        if fail_third and promotion_calls == 3:
            raise IrisMemoryError("injected promotion failure")
        return original_promote(
            candidate_id, current_namespace, kind=kind, actor=actor, reason=reason
        )

    monkeypatch.setattr(mirror, "rebuild_from_store", rebuild)
    monkeypatch.setattr(store, "promote_candidate", promote)

    if fail_third:
        with pytest.raises(IrisMemoryError, match="injected promotion failure"):
            orchestrator.process_candidates(namespace)
    else:
        assert len(orchestrator.process_candidates(namespace)) == 3

    committed_count = 2 if fail_third else 3
    assert rebuild_item_counts == [committed_count]
    items = service.list_items([namespace])
    assert len(items) == committed_count
    content = target.read_text(encoding="utf-8")
    assert content.startswith("manual note\n")
    assert all(item.text in content for item in items)
    events = service.list_events(namespace)
    assert sum(event.event_type == MemoryEventType.ADD for event in events) == committed_count
    assert (
        sum(event.event_type == MemoryEventType.CANDIDATE_ACCEPT for event in events)
        == committed_count
    )
    accepted = service.list_candidates(namespace, status=MemoryCandidateStatus.ACCEPTED)
    assert len(accepted) == committed_count


def test_empty_candidate_batch_does_not_rebuild_mirror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"), mirror=mirror)

    def rebuild(store: MemoryStore, namespace: str) -> None:
        pytest.fail("an empty batch must not rebuild the mirror")

    monkeypatch.setattr(mirror, "rebuild_from_store", rebuild)
    assert MemoryOrchestrator(service).process_candidates("project") == []


def test_default_orchestrator_records_episode_without_extracting_candidates(tmp_path: Path) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"))
    assert MemoryOrchestrator(service).observe(MemoryObserveInput(text="观察事实")) == []
    assert service.list_candidates("project") == []
    assert service.list_items(["project"]) == []
    events = service.list_events("project")
    assert len(events) == 1
    assert events[0].event_type == MemoryEventType.OBSERVE
    assert events[0].episode_id is not None


@pytest.mark.parametrize("promotion_fails", [False, True])
def test_mirror_failure_preserves_batch_success_or_original_promotion_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    promotion_fails: bool,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(store, mirror=mirror)
    orchestrator = MemoryOrchestrator(service, extractor=RuleMemoryExtractor())
    for index in range(2):
        orchestrator.observe(MemoryObserveInput(text=f"candidate {index}"))
    original_promote = store.promote_candidate
    failure = IrisMemoryError("promotion failure")
    promoted = 0

    def promote(
        candidate_id: str, namespace: str, *, kind: MemoryItemKind, actor: MemoryActor, reason: str
    ) -> MemoryItem:
        nonlocal promoted
        if promotion_fails and promoted == 1:
            raise failure
        item = original_promote(candidate_id, namespace, kind=kind, actor=actor, reason=reason)
        promoted += 1
        return item

    def rebuild(current_store: MemoryStore, namespace: str) -> None:
        raise IrisMemoryError("mirror failure")

    monkeypatch.setattr(store, "promote_candidate", promote)
    monkeypatch.setattr(mirror, "rebuild_from_store", rebuild)
    with caplog.at_level("WARNING", logger="iris.memory.service"):
        if promotion_fails:
            with pytest.raises(IrisMemoryError) as caught:
                orchestrator.process_candidates("project")
            assert caught.value is failure
        else:
            assert len(orchestrator.process_candidates("project")) == 2
    assert len(service.list_items(["project"])) == (1 if promotion_fails else 2)
    assert len(caplog.records) == 1
