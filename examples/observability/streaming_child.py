"""父级流式响应、子 Agent 等待和恢复，以及一次可定位的模型失败。"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

if not __package__:
    # 同时支持文档中的文件路径入口和 Python 包导入。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.observability"

from iris.agents import AgentConfig
from iris.exceptions import IrisProviderError, IrisRunStateError
from iris.harness import AgentRunner, LiveFact, LivePublisher
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import AgentRunRequest, RunResult
from iris.message import (
    LLMResponse,
    ModelBlockDelta,
    ModelResponseCompleted,
    TextBlock,
    ToolUseBlock,
)
from iris.observability import AgentObservabilityConfig
from iris.observability.service import Observability
from iris.providers import CompletionProvider
from iris.runtime import RuntimeStreamEvent

from ._scripted import ScriptedProvider, configure_cli


class ConsolePublisher:
    """只把真实 live plane 中的父级文本增量写到终端。"""

    def publish(self, fact: LiveFact) -> None:
        """正常 terminal 只补换行；运行摘要由示例入口输出。"""
        if isinstance(fact, RuntimeStreamEvent) and fact.kind == "model.event":
            event = fact.model_event
            if isinstance(event, ModelBlockDelta) and event.channel == "text":
                print(event.delta, end="", flush=True)
            elif isinstance(event, ModelResponseCompleted):
                print()


async def run_example(
    workspace: Path,
    *,
    observability: Observability | None = None,
    capture_config: AgentObservabilityConfig | None = None,
    live_publisher: LivePublisher | None = None,
) -> tuple[RunResult, RunResult, RunResult]:
    """返回等待、恢复完成和独立失败三份 durable 结果，不跳过子调用协议。"""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    child_path = workspace / "child.yaml"
    child_path.write_text(
        yaml.safe_dump(
            {
                "name": "observability-child",
                "model": "openai/demo-child",
                "system": "先询问偏好，再用中文完成子任务。",
                "permissions": {"workspace": str(workspace)},
                "context_policy": {"enabled": False},
                "tools": {"builtin": ["human.ask"]},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    catalog = workspace / "subagents.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "default": "researcher",
                "agents": {
                    "researcher": {"path": "child.yaml", "description": "询问表达偏好并完成子任务"}
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    child = ScriptedProvider(
        [
            LLMResponse(
                provider="scripted",
                model="demo-child",
                id="child-question",
                content=[
                    ToolUseBlock(
                        id="ask-style",
                        name="ask_question",
                        input={"question": "希望简洁还是详细？"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                provider="scripted",
                model="demo-child",
                id="child-answer",
                content=[TextBlock(text="已按简洁风格完成子任务。")],
                finish_reason="stop",
            ),
        ]
    )

    def child_provider(config: AgentConfig, *, config_path: Path) -> CompletionProvider:
        """等待后重建 child runner 时继续同一离线脚本的下一步。"""
        return child

    parent = ScriptedProvider(
        [
            LLMResponse(
                provider="scripted",
                model="demo-parent",
                id="parent-delegate",
                content=[
                    ToolUseBlock(
                        id="delegate",
                        name="subagent",
                        input={"prompt": "确认用户偏好，再完成一句话说明。"},
                    )
                ],
                finish_reason="tool_calls",
                input_tokens=0,
            ),
            LLMResponse(
                provider="scripted",
                model="demo-parent",
                id="parent-answer",
                content=[
                    TextBlock(text="子任务已恢复并完成：观测让每次模型和工具调用都有迹可循。")
                ],
                finish_reason="stop",
                output_tokens=6,
            ),
            IrisProviderError("演示：模型服务暂时不可用。", usage={"output_tokens": 0}),
        ]
    )
    config = AgentConfig(
        name="observability-parent",
        model="openai/demo-parent",
        system="把说明任务委派给子 Agent。",
        permissions={"workspace": str(workspace)},
        tools={"subagent": str(catalog)},
        context_policy={"enabled": False},
        observability=capture_config
        or AgentObservabilityConfig(enabled=True, capture_content=True),
    )
    runner = AgentRunner.from_config(
        config,
        provider=parent,
        child_provider_factory=child_provider,
        observability=observability,
        live_publisher=live_publisher if live_publisher is not None else ConsolePublisher(),
    )
    try:
        waiting = await runner.start(
            AgentRunRequest(input="请委派子任务并确认我的表达偏好。", session_id="streaming-child")
        )
        interaction = waiting.pending_interaction
        if interaction is None:
            raise IrisRunStateError("示例预期子 Agent 提出问题，但未得到等待结果。")
        completed = await runner.resume(
            waiting.run.run_id,
            interaction_id=interaction.interaction_id,
            response=QuestionInteractionResponse(answer="简洁。"),
        )
        failed = await runner.start(
            AgentRunRequest(input="再演示一次模型调用失败。", session_id="failure")
        )
        return waiting, completed, failed
    finally:
        await runner.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """导出等待前后各 activation 与实际失败，最后由 runner 关闭自建服务。"""
    workspace, capture = configure_cli("streaming-child", __doc__ or "流式子 Agent 观测示例", argv)
    waiting, completed, failed = asyncio.run(run_example(workspace, capture_config=capture))
    print(f"工作目录：{workspace}")
    print(f"等待并恢复的运行：{waiting.run.run_id} → {completed.run.stop_reason.value}")
    print(f"失败示例运行：{failed.run.run_id} → {failed.run.stop_reason.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
