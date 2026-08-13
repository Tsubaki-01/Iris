"""恢复 waiting lifecycle run 的公共 SDK 示例。

命令入口将权限决定或问题答案映射为 typed HITL response。

Example:
    python -m examples.lifecycle.resume --help
"""

# region imports
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from iris.harness import AgentRunner, RunResult
from iris.hitl import PermissionInteractionResponse, QuestionInteractionResponse

from ._runner import build_runner

# endregion


async def resume_run(
    runner: AgentRunner,
    *,
    run_id: str,
    interaction_id: str,
    decision: Literal["approve", "reject"] | None,
    answer: str | None,
) -> RunResult:
    """以 typed 人工响应恢复指定的 waiting run。

    Args:
        runner (AgentRunner): 已装配 provider 的 lifecycle runner。
        run_id (str): waiting logical run 的稳定 identity。
        interaction_id (str): 要消费的 pending interaction identity。
        decision (Literal["approve", "reject"] | None): 权限 interaction 的决定。
        answer (str | None): 问题 interaction 的文本回答。

    Returns:
        RunResult: resume 后已持久化的 waiting 或 terminal run 结果。
    """
    response = (
        PermissionInteractionResponse(decision=decision)
        if decision is not None
        else QuestionInteractionResponse(answer=answer or "")
    )
    return await runner.resume(
        run_id,
        interaction_id=interaction_id,
        response=response,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数并恢复 waiting run。

    Args:
        argv (Sequence[str] | None): 可选参数序列；为 ``None`` 时读取进程参数。

    Returns:
        int: SDK 调用成功后返回 ``0``；argparse 的 help 或参数错误会在此之前终止。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("agent.yaml"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--interaction-id", required=True)
    response_group = parser.add_mutually_exclusive_group(required=True)
    response_group.add_argument("--decision", choices=("approve", "reject"))
    response_group.add_argument("--answer")
    args = parser.parse_args(argv)

    runner = build_runner(
        args.config,
        env_file=args.env_file,
        requires_provider=True,
    )
    result = asyncio.run(
        resume_run(
            runner,
            run_id=args.run_id,
            interaction_id=args.interaction_id,
            decision=args.decision,
            answer=args.answer,
        )
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
