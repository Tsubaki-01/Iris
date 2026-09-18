"""让真实聊天模型调用 Web 工具，并将完整 run 保存到 SQLite。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from iris.config import init_config, is_config_initialized
from iris.harness import AgentRunner, AgentRunOptions, AgentRunRequest, RunLimits, RunResult
from iris.lifecycle import RunStopReason

CONFIG_PATH = Path(__file__).with_name("agent.yaml")
DEFAULT_PROMPT = (
    "请搜索 Python 官方文档，限定 docs.python.org，最多返回 3 条来源；"
    "再读取搜索结果中相关页面的摘录。解释 asyncio.TaskGroup 中一个任务失败时，"
    "其他任务和 ExceptionGroup 如何处理，并附来源链接。"
)


async def run_agent(*, config_path: Path, prompt: str, session_id: str) -> RunResult:
    """装配 YAML 中的真实 provider，完成一次 run 后关闭 runner。

    Args:
        config_path: Agent YAML 的路径。
        prompt: 要交给模型的联网问题。
        session_id: 持久化会话标识；复用它可以继续之前的对话。

    Returns:
        持久化的运行结果，包含状态、最终回答或错误。
    """
    runner = AgentRunner.from_config_path(config_path)
    try:
        return await runner.start(
            AgentRunRequest(input=prompt, session_id=session_id),
            options=AgentRunOptions(limits=RunLimits(max_model_steps=8)),
        )
    finally:
        await runner.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """执行真实模型与 Tavily 调用，输出 RunResult JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--session-id", default=f"web-{uuid4().hex}")
    args = parser.parse_args(argv)
    if not is_config_initialized():
        init_config(env_file=str(args.env_file) if args.env_file is not None else None)
    result = asyncio.run(
        run_agent(
            config_path=args.config,
            prompt=args.prompt,
            session_id=args.session_id,
        )
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.run.stop_reason is RunStopReason.COMPLETED else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
