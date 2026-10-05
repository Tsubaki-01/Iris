"""实际 Memory 写工具经 Run 捕获关联后遵守单会话 WAITING 资格。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.harness import AgentRunner, MaintenanceCoordinator, MemoryMaintenanceBinding
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import AgentRunRequest, RunPhase
from iris.memory import MemoryWriteInput
from iris.memory.generation_models import DreamPlan, GenerationResult
from iris.message import LLMRequest, LLMResponse, ToolUseBlock

from .fakes import StaticProvider, text_response, tool_response
from .test_maintenance_coordinator import memory_service


class SourceGenerationProvider(StaticProvider):
    """以空操作消费合格来源，概览修复与学习请求分别计数。"""

    def __init__(self) -> None:
        super().__init__()
        self.dream_requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """依真实输入形状返回该领域响应，不伪造工具或捕获记录。"""
        self.requests.append(request)
        data = json.loads(request.messages[1].text)
        if "groups" in data:
            return text_response('{"core_facts":"", "knowledge_scope":"项目工作偏好"}')
        if "records" in data:
            return text_response('{"observations":[]}')
        self.dream_requests.append(request)
        return text_response('{"operations":[], "resolutions":[]}')


async def wait_for_maintenance(coordinator: MaintenanceCoordinator) -> None:
    """等到本轮真实调度与捕获通知收口，不以固定 sleep 猜模型是否执行。"""
    async with asyncio.timeout(5):
        while coordinator._timer is not None or coordinator._task is not None:
            await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["remember", "update", "forget"])
async def test_memory_tool_changes_wait_for_source_run_to_finish(
    tmp_path: Path, operation: str
) -> None:
    """真实写入事件留在 WAITING 来源，终态后自动整理一次。"""
    generation = SourceGenerationProvider()
    service = memory_service(tmp_path / "memory.db", generation)
    params: dict[str, object] = {"reason": "用户明确说明"}
    if operation == "remember":
        params["text"] = "用户偏好中文回答"
    else:
        item = service.remember(MemoryWriteInput(text="原偏好", reason="已有知识"))
        service.store.commit_dream(
            service.store.read_dream_snapshot("project"),
            DreamPlan(),
            result=GenerationResult(namespace="project", stage="dream", status="completed"),
        )
        params["item_id"] = item.id
        if operation == "update":
            params["patch"] = {"text": "用户偏好中文回答"}
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="memory-call", name=f"memory_{operation}", input=params)),
        tool_response(
            ToolUseBlock(id="question", name="ask_question", input={"question": "继续？"})
        ),
        text_response("已完成"),
    )
    config = AgentConfig.model_validate(
        {
            "name": "writer",
            "model": "openai/test",
            "system": "帮助用户",
            "context_policy": {"enabled": False},
            "permissions": {"workspace": str(tmp_path), "writes": "allow"},
            "memory": {"enabled": True, "generation": {"enabled": True}},
            "tools": {"builtin": [f"memory.{operation}", "human.ask"]},
        }
    )
    runner = AgentRunner.from_config(config, provider=provider, memory_service=service)
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    runner.bind_maintenance(
        coordinator,
        memory=MemoryMaintenanceBinding(
            service=service,
            database_path=tmp_path / "memory.db",
            namespace="project",
        ),
    )
    try:
        waiting = await runner.start(AgentRunRequest(input="保存我的偏好", session_id="a"))
        assert waiting.run.phase is RunPhase.WAITING
        await wait_for_maintenance(coordinator)
        assert service.generation_state("project").pending_changes == 1
        assert generation.dream_requests == []
        sources = service.list_pending_sources("project")
        assert any(
            source.run_id == waiting.run.run_id and source.session_id == "a" for source in sources
        )

        completed = await runner.resume(
            waiting.run.run_id,
            interaction_id=waiting.pending_interaction.interaction_id,
            response=QuestionInteractionResponse(answer="继续"),
        )
        assert completed.run.phase is RunPhase.TERMINAL
        await wait_for_maintenance(coordinator)
        assert service.generation_state("project").pending_changes == 0
        assert len(generation.dream_requests) == 1
    finally:
        await coordinator.aclose()
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("same_run_id", [False, True])
async def test_repeated_tool_call_ids_do_not_mix_independent_sources(
    tmp_path: Path, same_run_id: bool
) -> None:
    """另一个来源的同名调用正在 WAITING，不能阻止 A 的合格工具变更。"""
    generation = SourceGenerationProvider()
    service = memory_service(tmp_path / "memory.db", generation)
    config = AgentConfig.model_validate(
        {
            "name": "writer",
            "model": "openai/test",
            "system": "帮助用户",
            "context_policy": {"enabled": False},
            "permissions": {"workspace": str(tmp_path), "writes": "allow"},
            "memory": {"enabled": True, "generation": {"enabled": True}},
            "tools": {"builtin": ["memory.remember", "human.ask"]},
        }
    )
    first = AgentRunner.from_config(
        config,
        memory_service=service,
        provider=StaticProvider(
            tool_response(
                ToolUseBlock(
                    id="call_1",
                    name="memory_remember",
                    input={
                        "text": "A 的事实",
                        "reason": "用户明确说明",
                    },
                )
            ),
            text_response("A 完成"),
        ),
    )
    second = AgentRunner.from_config(
        config,
        memory_service=service,
        provider=StaticProvider(
            tool_response(
                ToolUseBlock(
                    id="call_1",
                    name="memory_remember",
                    input={
                        "text": "B 的事实",
                        "reason": "用户明确说明",
                    },
                )
            ),
            tool_response(
                ToolUseBlock(id="ask", name="ask_question", input={"question": "继续？"})
            ),
            text_response("B 完成"),
        ),
    )
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    binding = MemoryMaintenanceBinding(
        service=service,
        database_path=tmp_path / "memory.db",
        namespace="project",
    )
    first.bind_maintenance(coordinator, memory=binding)
    second.bind_maintenance(coordinator, memory=binding)
    coordinator._foreground_enter()
    try:
        await first.start(AgentRunRequest(input="A", session_id="shared", run_id="run-a"))
        waiting = await second.start(
            AgentRunRequest(
                input="B",
                session_id="shared",
                run_id="run-a" if same_run_id else "run-b",
            )
        )
    finally:
        coordinator._foreground_exit()
    try:
        assert waiting.run.phase is RunPhase.WAITING
        assert first.store.source_id != second.store.source_id
        await wait_for_maintenance(coordinator)
        assert len(generation.dream_requests) == 1
        assert service.generation_state("project").pending_changes == 1
        await second.resume(
            waiting.run.run_id,
            interaction_id=waiting.pending_interaction.interaction_id,
            response=QuestionInteractionResponse(answer="继续"),
        )
        await wait_for_maintenance(coordinator)
        assert len(generation.dream_requests) == 2
        assert service.generation_state("project").pending_changes == 0
    finally:
        await coordinator.aclose()
        await first.aclose()
        await second.aclose()
