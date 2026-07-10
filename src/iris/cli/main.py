"""Iris CLI 入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .chat import ChatOptions, run_chat


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数并执行对应命令。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "chat":
        return run_chat(
            ChatOptions(
                config_path=Path(args.agent_config),
                session_id=args.session_id,
                max_steps=args.max_steps,
                trace_mode=args.trace,
                trace_file=Path(args.trace_file) if args.trace_file else None,
                env_file=Path(args.env_file) if args.env_file else None,
                include_tools=not args.no_tools,
            )
        )
    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    """构造 CLI parser。"""
    parser = argparse.ArgumentParser(prog="iris")
    subparsers = parser.add_subparsers(dest="command")

    chat = subparsers.add_parser("chat", help="启动多轮交互式 Agent chat")
    chat.add_argument("agent_config", help="agent.yaml 路径")
    chat.add_argument("--session-id", default="cli", help="会话 ID")
    chat.add_argument("--max-steps", type=int, default=8, help="每轮最大 provider 步数")
    chat.add_argument(
        "--trace",
        choices=("off", "compact", "full"),
        default="compact",
        help="request/response trace 展示模式",
    )
    chat.add_argument("--trace-file", default=None, help="可选 JSONL trace 输出路径")
    chat.add_argument("--env-file", default=None, help="可选 dotenv 文件路径")
    chat.add_argument("--no-tools", action="store_true", help="不向 provider 暴露工具")
    return parser


__all__ = ["main"]
