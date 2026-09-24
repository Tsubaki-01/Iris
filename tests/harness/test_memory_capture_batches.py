"""长 run 分批采集，页末不会误当成终态边界。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from iris.harness._memory_maintenance import MemoryMaintenance
from iris.lifecycle import FinishRun, RunStopReason
from iris.memory import MemoryEpisode, MemoryService, SQLiteMemoryStore
from iris.memory.generation_models import MemoryCaptureSource
from iris.message import Msg
from iris.store import InMemoryLifecycleStore, SQLiteStore

from ..store.test_run_input_commit import _input_command


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_long_terminal_capture_commits_pages_and_seals_only_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """260 条消息保存为三个有界 Episode，完整终点到达前不封源。"""
    lifecycle = (
        SQLiteStore(tmp_path / "lifecycle.db") if backend == "sqlite" else InMemoryLifecycleStore()
    )
    command = replace(
        _input_command(lifecycle),
        message_delta=[Msg.user(f"message {index}") for index in range(260)],
    )
    lifecycle.commit_run_input(command)
    run = lifecycle.load_run(command.run_id)
    assert run is not None
    lifecycle.finish_run(
        FinishRun(
            run_id=run.run_id,
            expected_run_revision=run.revision,
            activation_id=command.activation_id,
            stop_reason=RunStopReason.CANCELLED,
            now=command.now,
        )
    )
    memory = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"))
    maintenance = MemoryMaintenance(service=memory, namespace="project", lifecycle_store=lifecycle)
    pages: list[tuple[int, int | None]] = []
    original = memory.store.commit_capture

    def capture(
        source: MemoryCaptureSource,
        *,
        expected_captured_until: int,
        episode: MemoryEpisode | None,
    ) -> bool:
        pages.append((source.captured_until, source.terminal_message_count))
        return original(source, expected_captured_until=expected_captured_until, episode=episode)

    monkeypatch.setattr(memory.store, "commit_capture", capture)
    try:
        await maintenance.register_run(run)
        await maintenance.capture_pending()
        assert pages == [(128, None), (256, None), (260, 260)]
        episodes = sorted(
            (item.episode for item in memory.store.list_pending_episodes("project")),
            key=lambda episode: episode.metadata["start_message_count"],
        )
        assert [len(episode.records) for episode in episodes] == [128, 128, 4]
        assert [record.text for episode in episodes for record in episode.records] == [
            message.text for message in command.message_delta
        ]
        assert memory.store.list_capture_sources(lifecycle.source_id, "project") == []
        await maintenance.capture_pending()
        assert len(pages) == 3
    finally:
        await maintenance.aclose()
