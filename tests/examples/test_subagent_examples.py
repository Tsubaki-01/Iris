"""使用真实示例配置验证 Sub Agent 委派与跨 runner 问答恢复。"""

from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.harness import AgentRunner, ChildProviderFactory
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import AgentRunRequest, RunPhase, RunStopReason
from iris.message import ToolUseBlock
from iris.providers import CompletionProvider
from iris.store import SQLiteStore

from ..harness.fakes import StaticProvider, text_response, tool_response

CONFIG_PATH = Path(__file__).resolve().parents[2] / "examples/subagent/agent.yaml"


class _ChildProviders(ChildProviderFactory):
    """根据真实 child 配置选择离线 provider，保留其余运行链路。"""

    def __init__(self, **providers: StaticProvider) -> None:
        self.providers = providers

    def __call__(self, config: AgentConfig, *, config_path: Path) -> CompletionProvider:
        """使用已加载 child 的名称选择固定响应。"""
        return self.providers[config.name]


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", [None, "researcher"])
async def test_researcher_example_returns_child_result_to_parent(
    tmp_path: Path, selector: str | None
) -> None:
    """默认路由与显式路由都能加载 researcher 并把其结论交回父 run。"""
    params = {"prompt": "分析这段需求：Iris 面向 Python 开发者，提供 YAML 配置入口。"}
    if selector is not None:
        params["agent"] = selector
    parent = StaticProvider(
        tool_response(ToolUseBlock(id="delegate", name="subagent", input=params)),
        text_response("面向 Python 开发者提供 YAML 配置入口。"),
    )
    child = StaticProvider(text_response("目标用户：Python 开发者；入口：YAML。"))
    runner = AgentRunner.from_config_path(
        CONFIG_PATH,
        provider=parent,
        child_provider_factory=_ChildProviders(researcher=child),
        store=SQLiteStore(tmp_path / "subagent.db"),
    )

    result = await runner.start(AgentRunRequest(input="请委派分析需求。"))

    assert result.run.stop_reason == RunStopReason.COMPLETED, result.error
    assert result.assistant_message.text == "面向 Python 开发者提供 YAML 配置入口。"
    (call,) = runner.list_tool_calls(result.run.run_id)
    assert call.result.is_error is False
    assert call.result.model_content == "目标用户：Python 开发者；入口：YAML。"
    assert call.result.metadata["agent_selector"] == "researcher"
    child_run = runner.get_run(call.result.metadata["child_run_id"])
    assert child_run.stop_reason == RunStopReason.COMPLETED
    assert child_run.session_id != result.run.session_id
    assert any(
        block.content == "目标用户：Python 开发者；入口：YAML。"
        for message in parent.requests[-1].messages
        for block in message.tool_results
    )


@pytest.mark.asyncio
async def test_interviewer_example_resumes_through_parent_in_new_runner(tmp_path: Path) -> None:
    """子 Agent 提问后，新 runner 只用父 run 的 interaction 就能恢复并汇总结论。"""
    database = tmp_path / "subagent.db"
    runner = AgentRunner.from_config_path(
        CONFIG_PATH,
        provider=StaticProvider(
            tool_response(
                ToolUseBlock(
                    id="delegate",
                    name="subagent",
                    input={"agent": "interviewer", "prompt": "询问目标用户，再整理需求。"},
                )
            )
        ),
        child_provider_factory=_ChildProviders(
            interviewer=StaticProvider(
                tool_response(
                    ToolUseBlock(
                        id="ask-audience",
                        name="ask_question",
                        input={"question": "这个项目的目标用户是谁？"},
                    )
                )
            )
        ),
        store=SQLiteStore(database),
    )

    waiting = await runner.start(AgentRunRequest(input="请委派澄清需求。"))

    assert waiting.run.phase == RunPhase.WAITING, waiting.error
    assert waiting.pending_interaction.request.prompt.question == "这个项目的目标用户是谁？"
    resumed = AgentRunner.from_config_path(
        CONFIG_PATH,
        provider=StaticProvider(text_response("项目面向 Python 开发者。")),
        child_provider_factory=_ChildProviders(
            interviewer=StaticProvider(text_response("已确认：目标用户是 Python 开发者。"))
        ),
        store=SQLiteStore(database),
    )
    result = await resumed.resume(
        waiting.run.run_id,
        interaction_id=waiting.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="Python 开发者"),
    )

    assert result.run.stop_reason == RunStopReason.COMPLETED, result.error
    assert result.assistant_message.text == "项目面向 Python 开发者。"
    (call,) = resumed.list_tool_calls(result.run.run_id)
    assert call.result.is_error is False
    assert call.result.model_content == "已确认：目标用户是 Python 开发者。"
    assert call.result.metadata["agent_selector"] == "interviewer"
    (question,) = resumed.list_tool_calls(call.result.metadata["child_run_id"])
    assert "Python 开发者" in question.result.model_content
