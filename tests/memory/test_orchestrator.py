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
    MemoryScope,
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
    scope = MemoryScope(workspace_id="workspace", agent_id="agent")
    store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(store, mirror=mirror)
    orchestrator = MemoryOrchestrator(service, extractor=RuleMemoryExtractor())
    for index in range(3):
        orchestrator.observe(
            MemoryObserveInput(
                scope=scope, text=f"preference {index}", category=MemoryCategory.USER
            )
        )
    target = mirror.root / "User/user.md"
    target.write_text("manual note\n", encoding="utf-8")
    rebuild_item_counts: list[int] = []
    original_rebuild = mirror.rebuild_from_store
    original_promote = store.promote_candidate
    promotion_calls = 0

    def rebuild(current_store: MemoryStore, current_scope: MemoryScope) -> None:
        rebuild_item_counts.append(len(current_store.list_items(current_scope)))
        original_rebuild(current_store, current_scope)

    def promote(
        candidate_id: str,
        current_scope: MemoryScope,
        *,
        kind: MemoryItemKind,
        actor: MemoryActor,
        reason: str,
    ) -> MemoryItem:
        nonlocal promotion_calls
        promotion_calls += 1
        if fail_third and promotion_calls == 3:
            raise IrisMemoryError("injected promotion failure")
        return original_promote(candidate_id, current_scope, kind=kind, actor=actor, reason=reason)

    monkeypatch.setattr(mirror, "rebuild_from_store", rebuild)
    monkeypatch.setattr(store, "promote_candidate", promote)

    if fail_third:
        with pytest.raises(IrisMemoryError, match="injected promotion failure"):
            orchestrator.process_candidates(scope)
    else:
        assert len(orchestrator.process_candidates(scope)) == 3

    committed_count = 2 if fail_third else 3
    assert rebuild_item_counts == [committed_count]
    items = service.list_items(scope)
    assert len(items) == committed_count
    content = target.read_text(encoding="utf-8")
    assert content.startswith("manual note\n")
    assert all(item.text in content for item in items)
    events = service.list_events(scope)
    assert sum(event.event_type == MemoryEventType.ADD for event in events) == committed_count
    assert (
        sum(event.event_type == MemoryEventType.CANDIDATE_ACCEPT for event in events)
        == committed_count
    )
    accepted = service.list_candidates(scope, status=MemoryCandidateStatus.ACCEPTED)
    assert len(accepted) == committed_count


def test_empty_candidate_batch_does_not_rebuild_mirror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False), mirror=mirror)

    def rebuild(store: MemoryStore, scope: MemoryScope) -> None:
        pytest.fail("an empty batch must not rebuild the mirror")

    monkeypatch.setattr(mirror, "rebuild_from_store", rebuild)
    assert (
        MemoryOrchestrator(service).process_candidates(
            MemoryScope(workspace_id="workspace", agent_id="agent")
        )
        == []
    )
