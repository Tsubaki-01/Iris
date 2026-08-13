"""启动 lifecycle logical run 的公共 SDK 示例。

命令入口将请求映射到 ``AgentRunner.start()``。

Example:
    python -m examples.lifecycle.start --help
"""

# region imports
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from iris.harness import (
    AgentRunner,
    AgentRunOptions,
    AgentRunRequest,
    RunLimits,
    RunResult,
    RuntimeExecutionOptions,
)

from ._runner import build_runner

# endregion


async def start_run(
    runner: AgentRunner,
    *,
    input_text: str,
    session_id: str,
    run_id: str | None,
    max_steps: int,
    include_tools: bool,
) -> RunResult:
    """使用调用方提供的 runner 启动一个 logical run。

    Args:
        runner (AgentRunner): 已装配 provider 的 lifecycle runner。
        input_text (str): 本次 run 的初始用户输入。
        session_id (str): 归属 session 的稳定 identity。
        run_id (str | None): 可选的逻辑 run identity；为 ``None`` 时由 runner 生成。
        max_steps (int): 固定到本次 run 的最大 model step 数。
        include_tools (bool): 是否将配置的工具暴露给 runtime。

    Returns:
        RunResult: 已持久化的 waiting 或 terminal run 结果。
    """
    return await runner.start(
        AgentRunRequest(input=input_text, session_id=session_id, run_id=run_id),
        options=AgentRunOptions(
            limits=RunLimits(max_model_steps=max_steps),
            runtime=RuntimeExecutionOptions(include_tools=include_tools),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数并启动 run。

    Args:
        argv (Sequence[str] | None): 可选参数序列；为 ``None`` 时读取进程参数。

    Returns:
        int: SDK 调用成功后返回 ``0``；argparse 的 help 或参数错误会在此之前终止。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("agent.yaml"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--input", required=True)
    parser.add_argument("--session-id", default="example")
    parser.add_argument("--run-id")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--no-tools", action="store_true")
    args = parser.parse_args(argv)

    runner = build_runner(
        args.config,
        env_file=args.env_file,
        requires_provider=True,
    )
    result = asyncio.run(
        start_run(
            runner,
            input_text=args.input,
            session_id=args.session_id,
            run_id=args.run_id,
            max_steps=args.max_steps,
            include_tools=not args.no_tools,
        )
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
