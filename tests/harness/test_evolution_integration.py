"""真实 Run 经独立项目维护后由后续 Agent 使用经验。"""

from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.harness import AgentRunner, MaintenanceCoordinator
from iris.harness.evolution import build_project_evolution_binding
from iris.lifecycle import AgentRunRequest, RunStopReason
from iris.message import ToolUseBlock
from iris.prompts import PromptSource

from .fakes import StaticProvider, text_response, tool_response


@pytest.mark.asyncio
async def test_real_run_learns_skill_and_later_runner_reads_it(tmp_path: Path) -> None:
    """保存成功、catalog 发现与模型真正读到内容分别由完整链路证明。"""
    config = AgentConfig.model_validate(
        {
            "name": "learner",
            "model": "openai/test",
            "system": "遵循项目约定",
            "skills": {"enabled": True},
            "evolution": {"enabled": True},
            "permissions": {"workspace": str(tmp_path)},
            "context_policy": {"enabled": False},
        }
    )
    prompts = PromptSource.initialize(tmp_path)
    generation = StaticProvider(
        text_response('{"body":"使用 uv 运行精准测试。","reason":"用户约定"}')
    )
    binding = build_project_evolution_binding(
        config, workspace_root=tmp_path, prompt_source=prompts, provider=generation
    )
    coordinator = MaintenanceCoordinator(idle_seconds=1000)
    original = AgentRunner.from_config(
        config, prompt_source=prompts, provider=StaticProvider(text_response("已采用约定"))
    )
    original.bind_maintenance(coordinator, evolution=binding)
    later = None
    try:
        before = original.runtime.environment.skill_registry
        assert before is None or before.names() == ()
        completed = await original.start(AgentRunRequest(input="本项目使用 uv 运行精准测试"))
        assert completed.run.stop_reason is RunStopReason.COMPLETED
        result = await coordinator.request_project_experience(binding)
        assert result.status == "updated"
        assert "本项目使用 uv" in generation.requests[0].messages[1].text
        assert original.runtime.environment.skill_registry is before
        reader = StaticProvider(
            tool_response(
                ToolUseBlock(id="load", name="load_skill", input={"name": "project-experience"})
            ),
            text_response("将按已学习约定运行测试"),
        )
        later = AgentRunner.from_config(config, prompt_source=prompts, provider=reader)
        later.bind_maintenance(coordinator, evolution=binding)
        used = await later.start(AgentRunRequest(input="按项目约定工作", session_id="later"))
        assert used.run.stop_reason is RunStopReason.COMPLETED
        loaded = [
            result for message in reader.requests[1].messages for result in message.tool_results
        ]
        assert len(loaded) == 1 and not loaded[0].is_error
        assert "使用 uv 运行精准测试" in loaded[0].text
        assert len(generation.requests) == 1
    finally:
        if later is not None:
            await later.aclose()
        await original.aclose()
        await coordinator.aclose()
