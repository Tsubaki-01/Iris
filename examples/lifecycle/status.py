"""读取 lifecycle run durable 状态的公共 SDK 示例。

该入口只读取持久化状态，不发起 provider 请求。

Example:
    python -m examples.lifecycle.status --help
"""

# region imports
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from iris.harness import AgentRunner, RunResult, RunSnapshot

from ._runner import build_runner

# endregion


def read_status(runner: AgentRunner, *, run_id: str) -> RunSnapshot | RunResult:
    """读取 run snapshot，并优先返回 durable result。

    Args:
        runner (AgentRunner): 仅用于 durable read 的 provider-independent runner。
        run_id (str): 要查询的逻辑 run identity。

    Returns:
        RunSnapshot | RunResult: terminal/waiting 时的 durable result，否则为当前 snapshot。
    """
    snapshot = runner.get_run(run_id)
    return runner.get_result(run_id) or snapshot


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数并读取 run 状态。

    Args:
        argv (Sequence[str] | None): 可选参数序列；为 ``None`` 时读取进程参数。

    Returns:
        int: SDK 调用成功后返回 ``0``；argparse 的 help 或参数错误会在此之前终止。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("agent.yaml"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    runner = build_runner(
        args.config,
        env_file=args.env_file,
        requires_provider=False,
    )
    subject = read_status(runner, run_id=args.run_id)
    print(subject.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
