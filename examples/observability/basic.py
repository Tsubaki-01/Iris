"""离线模型驱动真实文件工具，产生模型→工具→模型调用树。"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

if not __package__:
    # 同时支持文档中的文件路径入口和 Python 包导入。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.observability"

from iris.agents import AgentConfig
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest, RunResult
from iris.message import LLMResponse, TextBlock, ToolUseBlock
from iris.observability import AgentObservabilityConfig
from iris.observability.service import Observability

from ._scripted import ScriptedProvider, configure_cli


async def run_example(
    workspace: Path,
    *,
    observability: Observability | None = None,
    capture_config: AgentObservabilityConfig | None = None,
) -> RunResult:
    """通过公开装配执行一次真实读取；注入的完整观测服务由调用方关闭。"""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "说明.txt").write_text(
        "Observability 记录 Agent 的模型和工具调用，帮助定位耗时、输入输出和失败位置。\n",
        encoding="utf-8",
    )
    provider = ScriptedProvider(
        [
            LLMResponse(
                provider="scripted",
                model="demo-basic",
                id="basic-read",
                content=[
                    ToolUseBlock(id="read-note", name="read_file", input={"file_path": "说明.txt"})
                ],
                finish_reason="tool_calls",
                input_tokens=0,
            ),
            LLMResponse(
                provider="scripted",
                model="demo-basic",
                id="basic-answer",
                content=[
                    TextBlock(text="观测记录模型和工具的调用，让我们看清执行过程、耗时与结果。")
                ],
                finish_reason="stop",
                output_tokens=6,
            ),
        ]
    )
    config = AgentConfig(
        name="observability-basic",
        model="openai/demo-basic",
        system="读取说明后用中文解释。",
        permissions={"workspace": str(workspace)},
        tools={"builtin": ["file.read"]},
        context_policy={"enabled": False},
        observability=capture_config
        or AgentObservabilityConfig(enabled=True, capture_content=True),
    )
    runner = AgentRunner.from_config(config, provider=provider, observability=observability)
    try:
        return await runner.start(
            AgentRunRequest(input="请读取说明.txt并解释观测的作用。", session_id="basic")
        )
    finally:
        await runner.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """配置 OTLP 并在 root runner 关闭时统一排空自有 exporter。"""
    workspace, capture = configure_cli("basic", __doc__ or "普通调用观测示例", argv)
    result = asyncio.run(run_example(workspace, capture_config=capture))
    print(
        f"工作目录：{workspace}\n运行 ID：{result.run.run_id}\n状态：{result.run.stop_reason.value}"
    )
    if result.assistant_message is not None:
        print(result.assistant_message.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
