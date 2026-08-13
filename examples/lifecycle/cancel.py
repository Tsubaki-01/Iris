"""取消 lifecycle run 的公共 SDK 示例。

该入口仅处理 durable lifecycle cancellation 与 settlement，不发起 provider 请求。

Example:
    python -m examples.lifecycle.cancel --help
"""

# region imports
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from iris.harness import AgentRunner, RunResult

from ._runner import build_runner

# endregion


async def cancel_run(
    runner: AgentRunner,
    *,
    run_id: str,
    reason: str | None,
    settlement_timeout: float | None,
) -> RunResult:
    """请求取消指定 run，并等待 durable settlement。

    Args:
        runner (AgentRunner): provider-independent lifecycle runner。
        run_id (str): 要取消的逻辑 run identity。
        reason (str | None): 可选的 durable cancellation 原因。
        settlement_timeout (float | None): 等待 durable terminal result 的秒数。

    Returns:
        RunResult: 已结算的 terminal run 结果。
    """
    return await runner.cancel(
        run_id,
        reason=reason,
        settlement_timeout=settlement_timeout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数并取消 run。

    Args:
        argv (Sequence[str] | None): 可选参数序列；为 ``None`` 时读取进程参数。

    Returns:
        int: SDK 调用成功后返回 ``0``；argparse 的 help 或参数错误会在此之前终止。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("agent.yaml"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reason")
    parser.add_argument("--settlement-timeout", type=float)
    args = parser.parse_args(argv)

    runner = build_runner(
        args.config,
        env_file=args.env_file,
        requires_provider=False,
    )
    result = asyncio.run(
        cancel_run(
            runner,
            run_id=args.run_id,
            reason=args.reason,
            settlement_timeout=args.settlement_timeout,
        )
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
