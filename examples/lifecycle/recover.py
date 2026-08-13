"""恢复 active lifecycle run 的公共 SDK 示例。

命令入口将精确 activation fence 传递给 ``AgentRunner.recover()``。

Example:
    python -m examples.lifecycle.recover --help
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


async def recover_run(
    runner: AgentRunner,
    *,
    run_id: str,
    activation_id: str,
) -> RunResult:
    """使用精确 activation fence 恢复 active run。

    Args:
        runner (AgentRunner): 已装配 provider 的 lifecycle runner。
        run_id (str): 要恢复的 active logical run identity。
        activation_id (str): 预期仍有效的 activation fence identity。

    Returns:
        RunResult: recover 后已持久化的 waiting 或 terminal run 结果。
    """
    return await runner.recover(run_id, expected_activation_id=activation_id)


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数并恢复 active run。

    Args:
        argv (Sequence[str] | None): 可选参数序列；为 ``None`` 时读取进程参数。

    Returns:
        int: SDK 调用成功后返回 ``0``；argparse 的 help 或参数错误会在此之前终止。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("agent.yaml"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--activation-id", required=True)
    args = parser.parse_args(argv)

    runner = build_runner(
        args.config,
        env_file=args.env_file,
        requires_provider=True,
    )
    result = asyncio.run(
        recover_run(
            runner,
            run_id=args.run_id,
            activation_id=args.activation_id,
        )
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
