"""Iris CLI 入口。

Example:
    exit_code = main(["chat", "agent.yaml"])
"""

# region imports
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .chat import ChatOptions, run_chat

# endregion


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数并执行 chat 命令。

    Args:
        argv (Sequence[str] | None): 可选命令行参数；为 ``None`` 时读取进程参数。

    Returns:
        int: 进程退出码。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    try:
        options = ChatOptions(
            config_path=Path(args.agent_config),
            session_id=args.session_id,
            max_steps=args.max_steps,
            env_file=Path(args.env_file) if args.env_file else None,
            include_tools=not args.no_tools,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return run_chat(options)


def _build_parser() -> argparse.ArgumentParser:
    """构造只包含 chat 子命令的参数解析器。

    Returns:
        argparse.ArgumentParser: 配置完成的参数解析器。
    """
    parser = argparse.ArgumentParser(prog="iris")
    subparsers = parser.add_subparsers(dest="command")

    chat = subparsers.add_parser("chat", help="启动多轮交互式 Agent chat")
    chat.add_argument("agent_config", help="agent.yaml 路径")
    chat.add_argument("--session-id", default="cli", help="会话 ID")
    chat.add_argument("--max-steps", type=int, default=8, help="每轮最大 provider 步数")
    chat.add_argument("--env-file", default=None, help="可选 dotenv 文件路径")
    chat.add_argument("--no-tools", action="store_true", help="不向 provider 暴露工具")
    return parser


__all__ = ["main"]
