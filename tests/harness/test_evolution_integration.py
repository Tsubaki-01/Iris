"""真实 Run 经独立项目维护后由后续 Agent 使用经验。"""

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from iris.agents import AgentConfig, load_agent_config
from iris.evolution.history import PublicationRecord
from iris.evolution.models import EvolutionResult, RevisionRequest, RevisionTarget
from iris.harness import AgentRunner, MaintenanceCoordinator
from iris.harness.evolution import build_project_evolution_binding
from iris.lifecycle import AgentRunRequest, RunStopReason
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
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


@pytest.mark.asyncio
async def test_config_revision_is_adopted_only_by_a_new_runner(tmp_path: Path) -> None:
    """显式 B 改写主文件，旧 runner 的新会话仍使用构造时配置。"""
    config_path = tmp_path / "agent.yaml"
    raw = {
        "name": "learner",
        "model": "openai/test",
        "system": "旧指令：详细回答",
        "skills": {"enabled": True},
        "evolution": {"enabled": True, "config_targets": ["system"]},
        "permissions": {"workspace": "."},
        "context_policy": {"enabled": False},
    }
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    config = load_agent_config(config_path)
    prompts = PromptSource.initialize(tmp_path)
    revision = StaticProvider(
        text_response(
            json.dumps(
                {
                    "action": "config",
                    "assignments": {"system": "新指令：只回答重点"},
                    "reason": "按明确请求精简",
                },
                ensure_ascii=False,
            )
        )
    )
    binding = build_project_evolution_binding(
        config,
        workspace_root=tmp_path,
        prompt_source=prompts,
        provider=revision,
        config_path=config_path,
    )
    coordinator = MaintenanceCoordinator(idle_seconds=1000)
    old_provider = StaticProvider(text_response("旧 runner 的新会话"))
    old = AgentRunner.from_config(
        config, config_path=config_path, provider=old_provider, prompt_source=prompts
    )
    old.bind_maintenance(coordinator, evolution=binding)
    new = None
    try:
        changed = await coordinator.request_revision(
            binding,
            RevisionRequest(
                description="把简单模式 system 精简为只回答重点",
                targets=(RevisionTarget(kind="config", name="system"),),
            ),
        )
        assert changed.status == "updated" and changed.stage == "revision"
        assert load_agent_config(config_path).system == "新指令：只回答重点"
        await old.start(AgentRunRequest(input="测试", session_id="another-session"))
        assert "旧指令：详细回答" in old_provider.requests[0].messages[0].text
        newer_provider = StaticProvider(text_response("新 runner"))
        new = AgentRunner.from_config_path(config_path, provider=newer_provider)
        new.bind_maintenance(coordinator, evolution=binding)
        await new.start(AgentRunRequest(input="测试", session_id="new-runner"))
        assert "新指令：只回答重点" in newer_provider.requests[0].messages[0].text
        assert len(revision.requests) == 1
        assert "未知" in revision.requests[0].messages[1].text
    finally:
        await coordinator.aclose()
        if new is not None:
            await new.aclose()
        await old.aclose()


@pytest.mark.asyncio
async def test_terminal_experience_automatically_produces_and_settles_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实用户纠正经 A 的原文引用产生 B，两个阶段各自完成且不自触发。"""
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: learner\nmodel: openai/test\nsystem: 详细回答\n"
        "skills:\n  enabled: true\ncontext_policy:\n  enabled: false\n"
        "evolution:\n  enabled: true\n  config_targets: [system]\n",
        encoding="utf-8",
    )
    config = load_agent_config(path)
    source = PromptSource.initialize(tmp_path)

    class RevisingProvider(StaticProvider):
        """A 绑定实际捕获的 user 原文，B 仅返回开放字段赋值。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                payload = json.loads(request.messages[1].text)
                record = next(
                    record
                    for item in payload["materials"]
                    for record in item["records"]
                    if record["role"] == "user"
                )
                output = {
                    "body": None,
                    "reason": "明确的行为纠正",
                    "issue": {
                        "description": "主指令要求详细回答，与用户希望简洁的要求不符",
                        "targets": [{"kind": "config", "name": "system"}],
                        "evidence": [{"ref": record["ref"], "quote": "请以后简洁回答"}],
                    },
                }
            else:
                assert len(self.requests) == 2
                output = {
                    "action": "config",
                    "assignments": {"system": "简洁回答"},
                    "reason": "采用纠正",
                }
            return text_response(json.dumps(output, ensure_ascii=False))

    provider = RevisingProvider()
    binding = build_project_evolution_binding(
        config, workspace_root=tmp_path, prompt_source=source, provider=provider, config_path=path
    )
    settled = asyncio.Event()
    loop = asyncio.get_running_loop()
    original_settle = binding.service.store.settle_publication

    def observe_settle(record: PublicationRecord) -> EvolutionResult:
        result = original_settle(record)
        if result.stage == "revision":
            loop.call_soon_threadsafe(settled.set)
        return result

    monkeypatch.setattr(binding.service.store, "settle_publication", observe_settle)
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    runner = AgentRunner.from_config(
        config,
        config_path=path,
        prompt_source=source,
        provider=StaticProvider(text_response("收到")),
    )
    runner.bind_maintenance(coordinator, evolution=binding)
    try:
        await runner.start(AgentRunRequest(input="请以后简洁回答", run_id="correction"))
        await asyncio.wait_for(settled.wait(), 3)
        await coordinator.aclose()
        assert len(provider.requests) == 2
        assert load_agent_config(path).system == "简洁回答"
        assert binding.service.store.list_pending_sources() == ()
        assert runner.runtime.environment.agent_config.system == "详细回答"
    finally:
        await coordinator.aclose()
        await runner.aclose()
