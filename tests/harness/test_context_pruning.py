"""通过真实 runner 验证历史减载不会缓存调用或破坏原文回读。"""

from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.harness import AgentRunner, AgentRunRequest
from iris.message import LLMRequest, Msg, ToolUseBlock
from iris.store import InMemoryLifecycleStore

from .fakes import StaticProvider, text_response, tool_response

_BODY = "观察材料。" * 500


class _PressureProvider(StaticProvider):
    """按完整长正文计主要成本，稳定地在第三份中型输出后触发压力。"""

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """整包包装也计入成本；预览不再计为完整长正文。"""
        full_bodies = sum(
            _BODY in result.content
            for message in request.messages
            for result in message.tool_results
        )
        return len(request.model_dump_json()) // 100 + full_bodies * 10000


def _results(messages: list[Msg]) -> list[str]:
    """按模型看到的原顺序提取工具正文。"""
    return [result.content for message in messages for result in message.tool_results]


@pytest.mark.asyncio
@pytest.mark.parametrize("changing", [False, True])
async def test_real_calls_keep_original_results_and_recall_pruned_body(
    tmp_path: Path, changing: bool
) -> None:
    """相同调用照常执行；重复折叠和不同结果短化均只改变模型视图。"""
    provider = _PressureProvider(
        *(tool_response(ToolUseBlock(id=f"probe-{index}", name="probe")) for index in range(3)),
        tool_response(
            ToolUseBlock(
                id="recall", name="context_read", input={"ref": "result:2:0", "limit": 8000}
            )
        ),
        text_response("finished"),
    )
    store = InMemoryLifecycleStore()
    runner = AgentRunner.from_config(
        AgentConfig.model_validate(
            {
                "name": "observer",
                "model": "openai/test",
                "system": "检查状态，保留原始证据。",
                "permissions": {"workspace": str(tmp_path)},
                "compaction": {"input_budget_tokens": 32000},
            }
        ),
        provider=provider,
        store=store,
    )
    outputs: list[str] = []

    def probe() -> str:
        """每次真正读取一次状态，即使参数与上次完全相同。"""
        version = len(outputs) + 1 if changing else 1
        output = f"版本 {version}\n{_BODY}"
        outputs.append(output)
        return output

    registry = runner.runtime.environment.tool_bridge.tool_view.registry
    tool = registry.register_function(probe)
    tool.definition.context_retention = "observation"
    try:
        result = await runner.start(AgentRunRequest(input="连续检查三次，再回读第一次结果"))
        assert result.run.stop_reason.value == "completed"
        assert result.assistant_message.text == "finished"
        assert len(outputs) == 3
        assert len(provider.requests) == 5
        assert result.run.usage.tool_calls_committed == 4
        assert result.run.usage.model_steps_committed == 5
        assert result.run.usage.total_tokens == 37
        assert result.run.usage.compaction.total_tokens == 0
        before_recall = _results(provider.requests[3].messages)
        assert len(before_recall) == 3
        assert before_recall[1:] == outputs[1:]
        assert len(before_recall[0]) < len(outputs[0])
        assert "result:2:0" in before_recall[0]
        if changing:
            assert "重复正文" not in before_recall[0]
        else:
            assert "result:6:0" in before_recall[0]

        records = runner.list_tool_calls(result.run.run_id)
        assert len(records) == 4
        assert [record.result.model_content for record in records[:3]] == outputs
        assert all(record.result.artifact is None for record in records)
        assert records[3].result.data["content"] == outputs[0]
        snapshot = runner.get_session(result.run.session_id)
        assert _results(list(snapshot.messages))[:3] == outputs
        assert snapshot.compaction is None
    finally:
        await runner.aclose()
