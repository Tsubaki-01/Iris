"""项目修订与 Memory 并行时按完整生成周期采用项目 prompt。"""

import asyncio
import json
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.evolution.models import RevisionRequest, RevisionTarget
from iris.harness import MaintenanceCoordinator, MemoryMaintenanceBinding
from iris.harness.evolution import build_project_evolution_binding
from iris.memory import FileMemoryMirror, MemoryObserveInput, MemoryService, SQLiteMemoryStore
from iris.memory.generation import _DreamResponse, _FlushResponse
from iris.memory.models import MemoryOverviewContent
from iris.message import LLMRequest, LLMResponse
from iris.prompts import PromptSource
from iris.store import InMemoryLifecycleStore

from ..memory.test_generation import Provider, flush_one
from ..memory.test_prompt_source import _seed
from .fakes import StaticProvider, text_response


@pytest.mark.asyncio
async def test_revision_during_memory_cycle_is_adopted_by_the_next_cycle(tmp_path: Path) -> None:
    """B 在 flush 等待模型时完成发布，当前 dream 仍使用周期开始时的正文。"""
    prompts = PromptSource.initialize(tmp_path)
    _seed(prompts, "old")
    entered = (asyncio.Event(), asyncio.Event())
    released = (asyncio.Event(), asyncio.Event())

    def respond(data: dict[str, object]) -> dict[str, object]:
        if "records" in data:
            return flush_one(data)
        if "observations" in data:
            observation = data["observations"][0]
            return {
                "operations": [
                    {
                        "action": "add",
                        "new_key": "learned",
                        "text": "本项目使用 uv",
                        "category": "reference",
                        "kind": "fact",
                        "reason": "约定",
                        "evidence": observation["evidence"],
                    }
                ],
                "resolutions": [
                    {
                        "observation_id": observation["id"],
                        "target_id": "learned",
                        "reason": "保留约定",
                    }
                ],
            }
        return {"core_facts": "使用 uv", "knowledge_scope": "项目工具约定"}

    class PausedMemoryProvider(Provider):
        """每轮 flush 的实际模型请求由测试屏障放行。"""

        cycle = 0

        async def complete(self, request: LLMRequest) -> LLMResponse:
            if "records" in json.loads(request.messages[1].text):
                index = self.cycle
                self.cycle += 1
                entered[index].set()
                await released[index].wait()
            return await super().complete(request)

    memory_provider = PausedMemoryProvider(respond)
    memory = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        mirror=FileMemoryMirror(tmp_path / "mirror"),
        prompt_source=prompts,
        generation_provider=memory_provider,
        generation_model="test",
        overview_provider=memory_provider,
        overview_model="test",
    )
    new_body = "new-dream: schema 仅允许伪造字段 fake"
    revision_provider = StaticProvider(
        text_response(
            json.dumps(
                {
                    "action": "prompt",
                    "target": "memory_dream",
                    "body": new_body,
                    "reason": "更新项目整理策略",
                },
                ensure_ascii=False,
            )
        )
    )
    config = AgentConfig.model_validate(
        {
            "name": "learner",
            "model": "openai/test",
            "system": "遵循项目约定",
            "skills": {"enabled": True},
            "evolution": {"enabled": True, "prompt_targets": ["memory_dream"]},
            "permissions": {"workspace": str(tmp_path)},
        }
    )
    binding = build_project_evolution_binding(
        config, workspace_root=tmp_path, prompt_source=prompts, provider=revision_provider
    )
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    coordinator._attach(
        MemoryMaintenanceBinding(
            service=memory, database_path=tmp_path / "memory.db", namespace="project"
        ),
        InMemoryLifecycleStore(),
        evolution=binding,
    )
    memory.observe(MemoryObserveInput(text="本项目使用 uv"))
    try:
        await coordinator.prepare()
        await asyncio.wait_for(entered[0].wait(), 2)
        first_cycle = coordinator._task
        assert first_cycle is not None and not first_cycle.done()
        revised = await asyncio.wait_for(
            coordinator.request_revision(
                binding,
                RevisionRequest(
                    description="更新 Memory dream 策略",
                    targets=(RevisionTarget(kind="prompt", name="memory_dream"),),
                ),
            ),
            2,
        )
        assert revised.stage == "revision" and revised.status == "updated"
        assert (prompts.root / "memory_dream.j2").read_text(encoding="utf-8") == new_body
        assert not released[0].is_set() and not first_cycle.done()
        released[0].set()
        await asyncio.wait_for(asyncio.shield(first_cycle), 2)
        assert len(memory_provider.requests) == 3
        for request, stage, response_model in zip(
            memory_provider.requests,
            ("flush", "dream", "overview"),
            (_FlushResponse, _DreamResponse, MemoryOverviewContent),
            strict=True,
        ):
            assert request.messages[0].text.startswith(f"old-{stage}")
            assert (
                json.dumps(response_model.model_json_schema(), ensure_ascii=False)
                in request.messages[0].text
            )

        memory.observe(MemoryObserveInput(text="第二轮仍使用 uv"))
        await asyncio.wait_for(entered[1].wait(), 2)
        second_cycle = coordinator._task
        assert second_cycle is not None
        released[1].set()
        await asyncio.wait_for(asyncio.shield(second_cycle), 2)
        assert len(memory_provider.requests) == 6
        next_dream = memory_provider.requests[4].messages[0].text
        assert next_dream.startswith(new_body)
        assert json.dumps(_DreamResponse.model_json_schema(), ensure_ascii=False) in next_dream
        assert len(revision_provider.requests) == 1
    finally:
        for release in released:
            release.set()
        await coordinator.aclose()
