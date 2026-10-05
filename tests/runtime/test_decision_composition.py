"""使用离线模型替身验证工具发现与记忆召回两个独立接点的组合。"""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from iris.agents import AgentConfig
from iris.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    ScoreAnswer,
    ScoreQuestion,
)
from iris.harness import AgentRunner, AgentRunRequest
from iris.lifecycle import RunStopReason
from iris.memory import FileMemoryMirror, MemoryService, MemoryWriteInput, SQLiteMemoryStore
from iris.message import ToolUseBlock
from iris.prompts import PromptSource

from ..harness.fakes import StaticProvider, text_response, tool_batch_response, tool_response


class RecordingEvaluator:
    """同一借用实例响应 Choice 和 Score，记录各接点真正发起的请求。"""

    def __init__(self) -> None:
        self.requests: list[DecisionRequest] = []

    async def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        """为唯一 deferred 候选和唯一记忆分别返回确定结果。"""
        self.requests.append(request)
        answers: dict[str, DecisionAnswer] = {}
        for key, question in request.questions.items():
            if isinstance(question, ChoiceQuestion):
                state = cast(dict[str, Any], request.state)
                assert {tool["name"] for tool in state["tools"].values()} == {"lookup"}
                answers[key] = ChoiceAnswer("c0", {"c0": 1.0}, 1.0)
            else:
                answers[key] = ScoreAnswer(3.0, {3: 1.0}, cast(ScoreQuestion, question).levels, 1.0)
        return DecisionResponse("typesafe", "jev-test", answers, DecisionUsage(10, 1))


@pytest.mark.asyncio
@pytest.mark.parametrize("discovery", [False, True])
@pytest.mark.parametrize("recall", [False, True])
async def test_independent_flags_share_one_evaluator_and_reuse_committed_history(
    tmp_path: Path, discovery: bool, recall: bool
) -> None:
    """eager 直接执行；同批两个 Search 分别受各自开关控制，后续轮次不重新判断。"""
    decision_path = tmp_path / "decision.yaml"
    decision_path.write_text(
        f"tools: {{discovery: {str(discovery).lower()}}}\n"
        f"memory: {{recall: {str(recall).lower()}}}\n",
        encoding="utf-8",
    )
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        mirror=FileMemoryMirror(tmp_path / "mirror"),
        overview_provider=StaticProvider(
            text_response('{"core_facts":"","knowledge_scope":"Deployment operations"}')
        ),
        overview_model="fake-model",
        prompt_source=PromptSource.initialize(tmp_path),
    )
    item = service.remember(
        MemoryWriteInput(text="deployment guide: restore the previous release", reason="组合测试")
    )
    await service.refresh_overview("project")
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="direct", name="eager")),
        tool_batch_response(
            ToolUseBlock(id="discover", name="tool_search", input={"queries": ["lookup"]}),
            ToolUseBlock(id="recall", name="memory_search", input={"query": "deployment"}),
        ),
        text_response("本轮完成"),
        text_response("下一轮完成"),
    )
    evaluator = RecordingEvaluator()
    runner = AgentRunner.from_config(
        AgentConfig.model_validate(
            {
                "name": "composed",
                "model": "openai/test",
                "system": "先直接使用 eager，再发现工具并读取记忆。",
                "permissions": {"workspace": str(tmp_path)},
                "context_policy": {"deferred_tools": True},
                "memory": {"enabled": True},
                "decision": {"path": str(decision_path)},
            }
        ),
        provider=provider,
        memory_service=service,
        decision_client=evaluator,
    )
    registry = runner.runtime.environment.tool_bridge.tool_view.registry
    executed: list[str] = []

    def eager() -> str:
        """已披露工具无需 Decision 即可直接执行。"""
        assert not evaluator.requests
        executed.append("eager")
        return "direct result"

    registry.register_function(eager)
    registry.register_function(lambda: "unused", name="lookup", deferred=True)
    try:
        assert runner.runtime.environment.decision_client is (
            evaluator if discovery or recall else None
        )
        assert registry.get("tool_search").decision_client is (evaluator if discovery else None)
        assert registry.get("memory_search").decision_client is (evaluator if recall else None)
        first = await runner.start(AgentRunRequest(input="处理", run_id="first"))
        assert first.run.stop_reason is RunStopReason.COMPLETED, first.error
        assert executed == ["eager"]
        assert "lookup" not in {tool.name for tool in provider.requests[0].tools}
        assert "lookup" in {tool.name for tool in provider.requests[2].tools}
        records = runner.list_tool_calls("first")
        assert [record.tool_name for record in records] == ["eager", "tool_search", "memory_search"]
        assert json.loads(records[1].result.model_content)["selections"] == [
            {"query": "lookup", "tool": "lookup"}
        ]
        assert json.loads(records[2].result.model_content)["items"][0]["item_id"] == item.id
        assert len(evaluator.requests) == int(discovery) + int(recall)
        assert sorted(
            question.type
            for request in evaluator.requests
            for question in request.questions.values()
        ) == (["choice"] if discovery else []) + (["score"] if recall else [])
        second = await runner.start(AgentRunRequest(input="沿用结果", run_id="second"))
        assert second.run.stop_reason is RunStopReason.COMPLETED, second.error
        assert len(evaluator.requests) == int(discovery) + int(recall)
        replayed = [
            result for message in provider.requests[-1].messages for result in message.tool_results
        ]
        assert [result.name for result in replayed] == ["eager", "tool_search", "memory_search"]
    finally:
        await runner.aclose()
