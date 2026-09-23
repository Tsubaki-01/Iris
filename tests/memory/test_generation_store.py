"""记忆生成的持久边界与原子提交测试。"""

from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory.generation_models import (
    DreamOperation,
    DreamPlan,
    EpisodeSlice,
    FlushCommit,
    GenerationResult,
    MemoryCaptureSource,
    ObservationResolution,
)
from iris.memory.models import (
    MemoryEpisode,
    MemoryEvent,
    MemoryEventType,
    MemoryEvidenceRef,
    MemoryItem,
    MemoryItemPatch,
    MemoryObservation,
    MemoryRecord,
    MemorySearchQuery,
    MemorySourceType,
)
from iris.memory.sqlite import SQLiteMemoryStore


def _episode(store: SQLiteMemoryStore) -> MemoryEpisode:
    episode = MemoryEpisode(records=(MemoryRecord(id="message-1", role="user", text="使用 uv"),))
    return store.add_episode(episode, event=MemoryEvent(event_type=MemoryEventType.OBSERVE))


def _flush(store: SQLiteMemoryStore, episode: MemoryEpisode) -> MemoryObservation:
    evidence = MemoryEvidenceRef(kind="episode", source_id=episode.id, record_id="message-1", end=5)
    observation = MemoryObservation(text="使用 uv", reason="项目约定", evidence=(evidence,))
    commit = FlushCommit(
        namespace="project",
        slices=(EpisodeSlice(episode.id, "message-1", 0, 5, "使用 uv"),),
        observations=(observation,),
        result=GenerationResult(namespace="project", stage="flush", status="completed"),
    )
    assert store.commit_flush(commit)
    assert not store.commit_flush(commit)
    return observation


def test_flush_consumes_once_without_publishing(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    episode = _episode(store)
    observation = _flush(store, episode)
    assert store.list_pending_episodes("project") == []
    assert store.list_observations("project")[0].observation == observation
    assert store.search(MemorySearchQuery(query="uv"), ["project"]).items == ()


def test_dream_commit_is_atomic_and_fenced(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    observation = _flush(store, _episode(store))
    snapshot = store.read_dream_snapshot("project")
    plan = DreamPlan(
        operations=(
            DreamOperation(
                action="add",
                new_id="uv-item",
                text="项目使用 uv",
                evidence=observation.evidence,
                reason="项目约定",
            ),
        ),
        resolutions=(
            ObservationResolution(observation_id=observation.id, item_id="uv-item", reason="新增"),
        ),
    )
    result = GenerationResult(namespace="project", stage="dream", status="completed")
    assert store.commit_dream(snapshot, plan, result=result)
    assert not store.commit_dream(snapshot, plan, result=result)
    assert store.get_item("uv-item", ["project"]).evidence == observation.evidence
    assert store.generation_state("project").pending_observations == 0


def test_explicit_text_update_replaces_current_support(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    event = MemoryEvent(event_type=MemoryEventType.ADD, source_id="call-before")
    item = store.add_item(MemoryItem(text="使用 Poetry"), event=event)
    correction = MemoryEvent(event_type=MemoryEventType.UPDATE, source_id="call-after")
    updated = store.update_item(
        item.id, "project", MemoryItemPatch(text="使用 uv"), event=correction
    )
    assert updated.evidence == (MemoryEvidenceRef(kind="event", source_id=correction.id),)
    assert store.list_events("project", item_id=item.id)[0].before["text"] == "使用 Poetry"
    snapshot = store.read_dream_snapshot("project")
    assert len(snapshot.changes) == 2
    assert snapshot.items[0].text == "使用 uv"


def test_capture_watermark_and_terminal_seal_are_atomic(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    source = MemoryCaptureSource(
        lifecycle_source_id="host",
        run_id="run",
        session_id="session",
        namespace="project",
        initial_message_count=3,
        captured_until=3,
    )
    assert store.register_source(source) == source
    episode = MemoryEpisode(records=(MemoryRecord(id="new-message", text="新增"),))
    captured = source.model_copy(update={"captured_until": 4})
    assert store.commit_capture(captured, expected_captured_until=3, episode=episode)
    assert not store.commit_capture(captured, expected_captured_until=3, episode=episode)
    assert len(store.list_pending_episodes("project")) == 1
    terminal = captured.model_copy(update={"terminal_message_count": 4, "outcome": "completed"})
    assert store.commit_capture(terminal, expected_captured_until=4, episode=None)
    assert store.list_capture_sources("host", "project") == []


def test_partial_flush_survives_reopen_and_empty_output_advances(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    episode = _episode(store)
    result = GenerationResult(namespace="project", stage="flush", status="completed")
    first = FlushCommit(
        "project", (EpisodeSlice(episode.id, "message-1", 0, 2, "使用"),), (), result
    )
    assert store.commit_flush(first)
    reopened = SQLiteMemoryStore(store.path)
    assert reopened.list_pending_episodes("project")[0].cursor.text_offset == 2
    assert not reopened.commit_flush(first)
    second = FlushCommit(
        "project",
        (EpisodeSlice(episode.id, "message-1", 2, 5, " uv"),),
        (),
        GenerationResult(namespace="project", stage="flush", status="completed"),
    )
    assert reopened.commit_flush(second)
    assert reopened.list_pending_episodes("project") == []


def test_dream_failure_rolls_back_previous_operations_and_inputs(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    observation = _flush(store, _episode(store))
    snapshot = store.read_dream_snapshot("project")
    plan = DreamPlan(
        operations=(
            DreamOperation(
                action="add",
                new_id="first",
                text="项目使用 uv",
                evidence=observation.evidence,
                reason="有效",
            ),
            DreamOperation(
                action="add",
                new_id="second",
                text="无效来源",
                evidence=(MemoryEvidenceRef(kind="event", source_id="missing"),),
                reason="失败",
            ),
        ),
        resolutions=(
            ObservationResolution(observation_id=observation.id, item_id="first", reason="写入"),
        ),
    )
    with pytest.raises(IrisMemoryError, match="证据不存在"):
        store.commit_dream(
            snapshot,
            plan,
            result=GenerationResult(namespace="project", stage="dream", status="completed"),
        )
    assert store.list_items(["project"]) == []
    assert store.generation_state("project").pending_observations == 1
    assert store.read_namespace_state("project").item_revision == 0


@pytest.mark.parametrize("correction", ["update", "forget"])
def test_explicit_correction_fences_dream_without_consuming_inputs(
    tmp_path: Path,
    correction: str,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    observation = _flush(store, _episode(store))
    item = store.add_item(
        MemoryItem(text="项目使用 Poetry"), event=MemoryEvent(event_type=MemoryEventType.ADD)
    )
    snapshot = store.read_dream_snapshot("project")
    if correction == "update":
        store.update_item(
            item.id,
            "project",
            MemoryItemPatch(text="项目改用 uv"),
            event=MemoryEvent(event_type=MemoryEventType.UPDATE),
        )
    else:
        assert store.delete_item(
            item.id,
            "project",
            event=MemoryEvent(event_type=MemoryEventType.DELETE),
        )
    plan = DreamPlan(
        resolutions=(ObservationResolution(observation_id=observation.id, reason="重复"),)
    )
    assert not store.commit_dream(
        snapshot,
        plan,
        result=GenerationResult(namespace="project", stage="dream", status="completed"),
    )
    assert store.generation_state("project").pending_observations == 1
    assert store.generation_state("project").pending_changes == 2


def test_blocked_only_reopens_for_its_dependency_or_changed_budget(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    _flush(store, _episode(store))
    item = store.add_item(
        MemoryItem(text="uv 约定"), event=MemoryEvent(event_type=MemoryEventType.ADD)
    )
    snapshot = store.read_dream_snapshot("project")
    assert store.block_dream(
        snapshot, reason="完整材料超限", budget=100, dependency_item_ids=[item.id]
    )
    store.add_item(MemoryItem(text="无关主题"), event=MemoryEvent(event_type=MemoryEventType.ADD))
    assert store.generation_state("project").blocked_observations == 1
    assert store.retry_blocked("project", budget=100) == 0
    store.update_item(
        item.id,
        "project",
        MemoryItemPatch(text="uv"),
        event=MemoryEvent(event_type=MemoryEventType.UPDATE),
    )
    assert store.generation_state("project").blocked_observations == 0
    assert store.generation_state("project").pending_observations == 1


def test_dream_snapshot_keeps_related_tombstone_and_blocked_correction(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    item = store.add_item(
        MemoryItem(text="使用 uv"), event=MemoryEvent(event_type=MemoryEventType.ADD)
    )
    store.delete_item(
        item.id, "project", event=MemoryEvent(event_type=MemoryEventType.DELETE, reason="用户忘记")
    )
    explicit = store.read_dream_snapshot("project")
    assert store.block_dream(explicit, reason="超限", budget=1, dependency_item_ids=[item.id])
    _flush(store, _episode(store))
    snapshot = store.read_dream_snapshot("project")
    assert snapshot.changes == ()
    assert [candidate.id for candidate in snapshot.items] == [item.id]
    assert any(event.event_type == MemoryEventType.DELETE for event in snapshot.events)


def test_explicit_change_snapshot_includes_related_existing_knowledge(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    existing = store.add_item(
        MemoryItem(text="project uv convention"), event=MemoryEvent(event_type=MemoryEventType.ADD)
    )
    first = store.read_dream_snapshot("project")
    assert store.commit_dream(
        first,
        DreamPlan(),
        result=GenerationResult(namespace="project", stage="dream", status="completed"),
    )
    added = store.add_item(
        MemoryItem(text="project uv convention duplicate"),
        event=MemoryEvent(event_type=MemoryEventType.ADD),
    )
    snapshot = store.read_dream_snapshot("project")
    assert len(snapshot.changes) == 1
    assert {item.id for item in snapshot.items} == {existing.id, added.id}


def test_dream_merge_keeps_identity_and_noop_does_not_advance_revision(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first = store.add_item(
        MemoryItem(text="uv convention"), event=MemoryEvent(event_type=MemoryEventType.ADD)
    )
    second = store.add_item(
        MemoryItem(text="uv convention repeated"), event=MemoryEvent(event_type=MemoryEventType.ADD)
    )
    snapshot = store.read_dream_snapshot("project")
    assert store.commit_dream(
        snapshot,
        DreamPlan(
            operations=(
                DreamOperation(
                    action="merge",
                    target_id=first.id,
                    merge_ids=(second.id,),
                    text="项目使用 uv",
                    evidence=first.evidence,
                    reason="合并重复约定",
                ),
            )
        ),
        result=GenerationResult(namespace="project", stage="dream", status="completed"),
    )
    assert store.get_item(first.id, ["project"]).text == "项目使用 uv"
    assert store.get_item(second.id, ["project"]) is None
    assert (
        next(
            item
            for item in store.list_items(["project"], include_deleted=True)
            if item.id == second.id
        ).superseded_by
        == first.id
    )
    assert store.generation_state("project").pending_changes == 0
    assert store.read_namespace_state("project").item_revision == 3
    observation = _flush(store, _episode(store))
    unchanged = store.read_dream_snapshot("project")
    assert store.commit_dream(
        unchanged,
        DreamPlan(
            resolutions=(
                ObservationResolution(
                    observation_id=observation.id, item_id=first.id, reason="已存在"
                ),
            )
        ),
        result=GenerationResult(namespace="project", stage="dream", status="completed"),
    )
    assert store.read_namespace_state("project").item_revision == 3
    assert store.generation_state("project").pending_observations == 0


def _paired_observations(store: SQLiteMemoryStore) -> tuple[MemoryObservation, MemoryObservation]:
    """同一次提炼产生共享证据、之后可以分批整理的两条观察。"""
    episode = MemoryEpisode(records=(MemoryRecord(id="m", text="abcdef"),))
    store.add_episode(episode, event=MemoryEvent(event_type=MemoryEventType.OBSERVE))
    evidence = MemoryEvidenceRef(kind="episode", source_id=episode.id, record_id="m", end=6)
    first = MemoryObservation(text="old-label", reason="材料", evidence=(evidence,))
    second = MemoryObservation(text="unrelated-spelling", reason="同源观察", evidence=(evidence,))
    assert store.commit_flush(
        FlushCommit(
            "project",
            (EpisodeSlice(episode.id, "m", 0, 6, "abcdef"),),
            (first, second),
            GenerationResult(namespace="project", stage="flush", status="completed"),
        )
    )
    return first, second


def test_historical_source_relation_survives_replaced_support(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first, second = _paired_observations(store)
    assert store.commit_dream(
        store.read_dream_snapshot("project", observation_ids=[first.id]),
        DreamPlan(
            operations=(
                DreamOperation(
                    action="add",
                    new_id="stable",
                    text="old-label",
                    evidence=first.evidence,
                    reason="建立",
                ),
            ),
            resolutions=(
                ObservationResolution(observation_id=first.id, item_id="stable", reason="建立"),
            ),
        ),
        result=GenerationResult(namespace="project", stage="dream", status="completed"),
    )
    store.update_item(
        "stable",
        "project",
        MemoryItemPatch(text="corrected-name"),
        event=MemoryEvent(event_type=MemoryEventType.UPDATE),
    )
    snapshot = store.read_dream_snapshot("project", observation_ids=[second.id], change_ids=[])
    assert {item.id for item in snapshot.items} == {"stable"}
    assert any(event.event_type == MemoryEventType.UPDATE for event in snapshot.events)


def test_early_episode_reads_later_terminal_outcome(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    source = MemoryCaptureSource(
        lifecycle_source_id="host",
        run_id="run",
        session_id="session",
        namespace="project",
        initial_message_count=0,
        captured_until=0,
    )
    store.register_source(source)
    episode = MemoryEpisode(
        source_type=MemorySourceType.TASK,
        source_id="run",
        records=(MemoryRecord(id="m", text="work"),),
        metadata={"lifecycle_source_id": "host"},
    )
    captured = source.model_copy(update={"captured_until": 1})
    assert store.commit_capture(captured, expected_captured_until=0, episode=episode)
    assert store.list_pending_episodes("project")[0].source_outcome is None
    terminal = captured.model_copy(update={"terminal_message_count": 1, "outcome": "failed"})
    assert store.commit_capture(terminal, expected_captured_until=1, episode=None)
    progress = store.list_pending_episodes("project")[0]
    assert progress.source_outcome == "failed"
    assert progress.episode == episode


def test_concurrent_dream_only_consumes_once(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    store = SQLiteMemoryStore(tmp_path / "memory.db")
    second = SQLiteMemoryStore(store.path)
    observation = _flush(store, _episode(store))
    snapshot = store.read_dream_snapshot("project")
    plan = DreamPlan(
        operations=(
            DreamOperation(
                action="add",
                new_id="one-item",
                text="uv",
                evidence=observation.evidence,
                reason="约定",
            ),
        ),
        resolutions=(
            ObservationResolution(observation_id=observation.id, item_id="one-item", reason="建立"),
        ),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [
            pool.submit(
                candidate.commit_dream,
                snapshot,
                plan,
                result=GenerationResult(namespace="project", stage="dream", status="completed"),
            )
            for candidate in (store, second)
        ]
        assert sorted(future.result(timeout=5) for future in outcomes) == [False, True]
    assert len(store.list_items(["project"])) == 1
    assert store.generation_state("project").pending_observations == 0


def test_history_target_follows_merge_to_current_keeper(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first, second = _paired_observations(store)
    assert store.commit_dream(
        store.read_dream_snapshot("project", observation_ids=[first.id]),
        DreamPlan(
            operations=(
                DreamOperation(
                    action="add",
                    new_id="old",
                    text="old-label",
                    evidence=first.evidence,
                    reason="建立",
                ),
            ),
            resolutions=(
                ObservationResolution(observation_id=first.id, item_id="old", reason="建立"),
            ),
        ),
        result=GenerationResult(namespace="project", stage="dream", status="completed"),
    )
    keeper = store.add_item(
        MemoryItem(text="old-label canonical"), event=MemoryEvent(event_type=MemoryEventType.ADD)
    )
    snapshot = store.read_dream_snapshot("project", observation_ids=[])
    assert store.commit_dream(
        snapshot,
        DreamPlan(
            operations=(
                DreamOperation(
                    action="merge",
                    target_id=keeper.id,
                    merge_ids=("old",),
                    text="canonical current fact",
                    evidence=keeper.evidence,
                    reason="统一",
                ),
            )
        ),
        result=GenerationResult(namespace="project", stage="dream", status="completed"),
    )
    history = next(
        state for state in store.list_observations("project") if state.observation.id == first.id
    )
    assert history.item_id == "old"
    fresh = store.read_dream_snapshot("project", observation_ids=[second.id])
    assert {item.id for item in fresh.items} == {"old", keeper.id}


def test_disjoint_record_spans_do_not_force_unrelated_history_into_snapshot(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    episode = MemoryEpisode(records=(MemoryRecord(id="m", text="abcdef"),))
    store.add_episode(episode, event=MemoryEvent(event_type=MemoryEventType.OBSERVE))
    first = MemoryObservation(
        text="apples",
        reason="材料",
        evidence=(MemoryEvidenceRef(kind="episode", source_id=episode.id, record_id="m", end=3),),
    )
    assert store.commit_flush(
        FlushCommit(
            "project",
            (EpisodeSlice(episode.id, "m", 0, 3, "abc"),),
            (first,),
            GenerationResult(namespace="project", stage="flush", status="completed"),
        )
    )
    assert store.commit_dream(
        store.read_dream_snapshot("project"),
        DreamPlan(
            operations=(
                DreamOperation(
                    action="add",
                    new_id="old",
                    text="apples",
                    evidence=first.evidence,
                    reason="建立",
                ),
            ),
            resolutions=(
                ObservationResolution(observation_id=first.id, item_id="old", reason="建立"),
            ),
        ),
        result=GenerationResult(namespace="project", stage="dream", status="completed"),
    )
    second = MemoryObservation(
        text="bananas",
        reason="另一事实",
        evidence=(
            MemoryEvidenceRef(kind="episode", source_id=episode.id, record_id="m", start=3, end=6),
        ),
    )
    assert store.commit_flush(
        FlushCommit(
            "project",
            (EpisodeSlice(episode.id, "m", 3, 6, "def"),),
            (second,),
            GenerationResult(namespace="project", stage="flush", status="completed"),
        )
    )
    assert store.read_dream_snapshot("project").items == ()


def test_invalid_metadata_rolls_back_through_domain_error(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    with pytest.raises(IrisMemoryError, match="JSON"):
        store.add_item(
            MemoryItem(text="valid text", metadata={"object": object()}),
            event=MemoryEvent(event_type=MemoryEventType.ADD),
        )
    assert store.list_items(["project"]) == []


def test_dream_delete_preserves_evidence_and_does_not_create_new_change(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    item = store.add_item(
        MemoryItem(text="obsolete convention"),
        event=MemoryEvent(event_type=MemoryEventType.ADD),
    )
    snapshot = store.read_dream_snapshot("project")
    assert store.commit_dream(
        snapshot,
        DreamPlan(
            operations=(DreamOperation(action="delete", target_id=item.id, reason="已明确失效"),)
        ),
        result=GenerationResult(namespace="project", stage="dream", status="completed"),
    )
    assert store.get_item(item.id, ["project"]) is None
    assert store.search(MemorySearchQuery(query="obsolete"), ["project"]).items == ()
    (retired,) = store.list_items(["project"], include_deleted=True)
    assert retired.id == item.id and retired.status.value == "deleted"
    assert retired.evidence == item.evidence
    assert retired.deleted_at is not None
    event = store.list_events("project", item_id=item.id)[0]
    assert event.event_type is MemoryEventType.DREAM
    assert event.before["status"] == "active" and event.after["status"] == "deleted"
    assert store.generation_state("project").pending_changes == 0
    assert store.read_namespace_state("project").item_revision == 2


def test_dream_only_consumes_explicit_changes_selected_in_snapshot(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first_event = MemoryEvent(event_type=MemoryEventType.ADD)
    second_event = MemoryEvent(event_type=MemoryEventType.ADD)
    store.add_item(MemoryItem(text="apples"), event=first_event)
    store.add_item(MemoryItem(text="bananas"), event=second_event)
    snapshot = store.read_dream_snapshot("project", change_ids=[first_event.id])
    assert [change.event_id for change in snapshot.changes] == [first_event.id]
    assert store.commit_dream(
        snapshot,
        DreamPlan(),
        result=GenerationResult(namespace="project", stage="dream", status="completed"),
    )
    remaining = store.read_dream_snapshot("project")
    assert [change.event_id for change in remaining.changes] == [second_event.id]
    assert store.generation_state("project").pending_changes == 1
    assert store.read_namespace_state("project").item_revision == 2


def test_two_observations_support_existing_item_without_new_identity(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    item = store.add_item(
        MemoryItem(text="项目使用 uv"),
        event=MemoryEvent(event_type=MemoryEventType.ADD),
    )
    first_record = MemoryRecord(id="user-message", role="user", text="本项目用 uv")
    second_record = MemoryRecord(id="tool-result", role="tool", text="uv 项目配置已存在")
    episode = MemoryEpisode(records=(first_record, second_record))
    store.add_episode(episode, event=MemoryEvent(event_type=MemoryEventType.OBSERVE))
    observations = tuple(
        MemoryObservation(
            text="项目使用 uv",
            reason="独立来源支持项目约定",
            evidence=(
                MemoryEvidenceRef(
                    kind="episode", source_id=episode.id, record_id=record.id, end=len(record.text)
                ),
            ),
        )
        for record in episode.records
    )
    assert store.commit_flush(
        FlushCommit(
            "project",
            tuple(
                EpisodeSlice(episode.id, record.id, 0, len(record.text), record.text)
                for record in episode.records
            ),
            observations,
            GenerationResult(namespace="project", stage="flush", status="completed"),
        )
    )
    current_evidence = (
        *item.evidence,
        *(ref for observation in observations for ref in observation.evidence),
    )
    snapshot = store.read_dream_snapshot("project")
    assert store.commit_dream(
        snapshot,
        DreamPlan(
            operations=(
                DreamOperation(
                    action="support",
                    target_id=item.id,
                    evidence=current_evidence,
                    reason="补齐两个独立来源",
                ),
            ),
            resolutions=tuple(
                ObservationResolution(
                    observation_id=observation.id, item_id=item.id, reason="归入既有约定"
                )
                for observation in observations
            ),
        ),
        result=GenerationResult(namespace="project", stage="dream", status="completed"),
    )
    (updated,) = store.list_items(["project"])
    assert updated.id == item.id and updated.text == item.text
    assert updated.evidence == current_evidence
    states = store.list_observations("project")
    assert {state.observation.id for state in states} == {
        observation.id for observation in observations
    }
    assert all(state.item_id == item.id and state.status == "processed" for state in states)
    assert {state.observation.evidence for state in states} == {
        observation.evidence for observation in observations
    }
