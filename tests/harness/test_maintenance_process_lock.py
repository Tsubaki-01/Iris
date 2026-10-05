"""真实独立进程通过共享维护入口竞争同一 Memory 资源。"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Any

import pytest

from iris.harness import MaintenanceCoordinator, MemoryMaintenanceBinding
from iris.memory import MemoryObserveInput, MemoryService, SQLiteMemoryStore
from iris.memory.generation_models import MemoryMaintenanceScope
from iris.memory.mirror import FileMemoryMirror
from iris.message import LLMRequest, LLMResponse
from iris.prompts import PromptSource

from .fakes import StaticProvider, text_response
from .test_maintenance_coordinator import configured_runner


def _maintain_process(
    directory: str,
    namespace: str,
    prepared: Event,
    generated: Event,
    release: Event,
    finished: Event,
    calls: Any,
    exit_during_generation: bool,
) -> None:
    """子进程使用真实 service/cycle，事件只控制 provider 等待和测试观察。"""

    async def main() -> None:
        path = Path(directory)
        completed = asyncio.Event()

        class Provider(StaticProvider):
            async def complete(self, request: LLMRequest) -> LLMResponse:
                with calls.get_lock():
                    calls.value += 1
                generated.set()
                assert await asyncio.to_thread(release.wait, 15)
                if exit_during_generation:
                    os._exit(0)
                return text_response('{"observations": []}')

        class ObservedService(MemoryService):
            async def maintain_cycle(self, ns: str, *, scope: MemoryMaintenanceScope) -> bool:
                more = await super().maintain_cycle(ns, scope=scope)
                if not more:
                    completed.set()
                return more

        provider = Provider()
        service = ObservedService(
            SQLiteMemoryStore(path / "memory.db"),
            prompt_source=PromptSource.initialize(path),
            mirror=FileMemoryMirror(path / "mirror", workspace_root=path),
            generation_provider=provider,
            generation_model="generation",
            overview_provider=provider,
            overview_model="overview",
        )
        runner = configured_runner(path, service)
        config = runner.runtime.environment.agent_config
        runner.runtime.environment.agent_config = config.model_copy(
            update={"memory": config.memory.model_copy(update={"write_namespace": namespace})}
        )
        coordinator = MaintenanceCoordinator(idle_seconds=0)
        runner.bind_maintenance(
            coordinator,
            memory=MemoryMaintenanceBinding(
                service=service, database_path=path / "memory.db", namespace=namespace
            ),
        )
        try:
            await runner.aprepare()
            prepared.set()
            await asyncio.wait_for(completed.wait(), 15)
            assert service.generation_state(namespace).pending_episodes == 0
            finished.set()
        finally:
            await runner.aclose()
            await coordinator.aclose()

    asyncio.run(main())


@pytest.mark.parametrize("scenario", ["same_namespace", "different_namespace", "owner_dies"])
def test_independent_processes_coordinate_actual_generation(tmp_path: Path, scenario: str) -> None:
    """同资源后继重读消费状态；不同 namespace 并行；进程死亡由 OS 释放锁。"""
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    service = MemoryService(store)
    service.observe(MemoryObserveInput(text="待整理", namespace="project"))
    second_namespace = "other" if scenario == "different_namespace" else "project"
    if second_namespace == "other":
        service.observe(MemoryObserveInput(text="另一空间", namespace="other"))
    context = multiprocessing.get_context("spawn")
    events = [tuple(context.Event() for _ in range(4)) for _ in range(2)]
    calls = context.Value("i", 0)
    processes = [
        context.Process(
            target=_maintain_process,
            args=(
                str(tmp_path),
                namespace,
                *events[index],
                calls,
                scenario == "owner_dies" and index == 0,
            ),
        )
        for index, namespace in enumerate(("project", second_namespace))
    ]
    first, second = processes
    try:
        first.start()
        assert events[0][0].wait(10)
        assert events[0][1].wait(10)
        second.start()
        assert events[1][0].wait(10)
        if scenario == "different_namespace":
            assert events[1][1].wait(5)
            assert calls.value == 2
        else:
            assert not events[1][1].wait(0.3)
            assert not events[1][3].is_set()
            assert calls.value == 1
        if scenario == "owner_dies":
            events[0][2].set()
            first.join(5)
            assert first.exitcode == 0
            assert events[1][1].wait(5)
        events[0][2].set()
        events[1][2].set()
        if scenario != "owner_dies":
            assert events[0][3].wait(10)
        assert events[1][3].wait(10)
        second.join(5)
        assert second.exitcode == 0
        if scenario != "owner_dies":
            first.join(5)
            assert first.exitcode == 0
        assert calls.value == (1 if scenario == "same_namespace" else 2)
    finally:
        for group in events:
            group[2].set()
        for process in processes:
            if process.pid is not None:
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
