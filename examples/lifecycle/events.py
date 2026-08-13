"""读取 lifecycle run durable 事件的公共 SDK 示例。

该入口从持久化 store 继续排他 sequence 游标，不发起 provider 请求。

Example:
    python -m examples.lifecycle.events --help
"""

# region imports
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from iris.harness import AgentRunner, RunEvent

from ._runner import build_runner

# endregion


def read_events(
    runner: AgentRunner,
    *,
    run_id: str,
    after_sequence: int,
) -> tuple[list[RunEvent], int]:
    """读取排他游标后的 durable 事件。

    Args:
        runner (AgentRunner): 仅用于 durable read 的 provider-independent runner。
        run_id (str): 要查询的逻辑 run identity。
        after_sequence (int): 排他事件游标，只返回 sequence 更大的事件。

    Returns:
        tuple[list[RunEvent], int]: 读取到的事件与下一次使用的排他游标。
    """
    events = runner.list_events(run_id, after_sequence=after_sequence)
    next_after_sequence = events[-1].sequence if events else after_sequence
    return events, next_after_sequence


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数并读取 run 事件。

    Args:
        argv (Sequence[str] | None): 可选参数序列；为 ``None`` 时读取进程参数。

    Returns:
        int: SDK 调用成功后返回 ``0``；argparse 的 help 或参数错误会在此之前终止。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("agent.yaml"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--after-sequence", type=int, default=0)
    args = parser.parse_args(argv)

    runner = build_runner(
        args.config,
        env_file=args.env_file,
        requires_provider=False,
    )
    events, next_after_sequence = read_events(
        runner,
        run_id=args.run_id,
        after_sequence=args.after_sequence,
    )
    payload = {
        "run_id": args.run_id,
        "after_sequence": args.after_sequence,
        "next_after_sequence": next_after_sequence,
        "events": [event.model_dump(mode="json") for event in events],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
