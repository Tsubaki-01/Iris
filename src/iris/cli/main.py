"""Iris CLI 入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .chat import ChatOptions, run_chat
from .run import (
    RunCancelOptions,
    RunEventsOptions,
    RunRecoverOptions,
    RunResumeOptions,
    RunStartOptions,
    RunStatusOptions,
    run_cancel,
    run_events,
    run_recover,
    run_resume,
    run_start,
    run_status,
)


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
    if args.command == "run":
        try:
            return _dispatch_run(args)
        except ValueError as exc:
            parser.error(str(exc))
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

    run = subparsers.add_parser("run", help="执行一次显式 Agent lifecycle 操作")
    actions = run.add_subparsers(dest="action", required=True)

    start = actions.add_parser("start", help="创建并推进一个 logical run")
    start.add_argument("agent_config", help="agent.yaml 路径")
    start.add_argument("--input", required=True, help="本次 run 的用户输入")
    start.add_argument("--session-id", default="cli", help="会话 ID")
    start.add_argument("--run-id", default=None, help="可选 logical run ID")
    start.add_argument("--max-steps", type=int, default=20, help="最大 provider 步数")
    start.add_argument("--no-tools", action="store_true", help="不向 provider 暴露工具")
    _add_run_output_arguments(start)

    status = actions.add_parser("status", help="读取 logical run 的 durable 状态")
    status.add_argument("agent_config", help="agent.yaml 路径")
    status.add_argument("--run-id", required=True, help="logical run ID")
    _add_run_output_arguments(status)

    events = actions.add_parser("events", help="读取 logical run 的 durable event 时间线")
    events.add_argument("agent_config", help="agent.yaml 路径")
    events.add_argument("--run-id", required=True, help="logical run ID")
    events.add_argument(
        "--after-sequence",
        type=int,
        default=0,
        help="只读取 sequence 严格大于该值的事件",
    )
    _add_run_output_arguments(events)

    resume = actions.add_parser("resume", help="提交一次 typed HITL response")
    resume.add_argument("agent_config", help="agent.yaml 路径")
    resume.add_argument("--run-id", required=True, help="logical run ID")
    resume.add_argument("--interaction-id", required=True, help="pending interaction ID")
    response = resume.add_mutually_exclusive_group(required=True)
    response.add_argument(
        "--decision",
        choices=("approve", "reject"),
        default=None,
        help="permission response",
    )
    response.add_argument("--answer", default=None, help="question response")
    _add_run_output_arguments(resume)

    cancel = actions.add_parser("cancel", help="请求取消并观察 terminal settlement")
    cancel.add_argument("agent_config", help="agent.yaml 路径")
    cancel.add_argument("--run-id", required=True, help="logical run ID")
    cancel.add_argument("--reason", default=None, help="可选取消原因")
    cancel.add_argument(
        "--settlement-timeout",
        type=float,
        default=None,
        help="等待 durable settlement 的秒数",
    )
    _add_run_output_arguments(cancel)

    recover = actions.add_parser("recover", help="使用 activation fence 恢复 active run")
    recover.add_argument("agent_config", help="agent.yaml 路径")
    recover.add_argument("--run-id", required=True, help="logical run ID")
    recover.add_argument("--activation-id", required=True, help="预期 activation ID")
    _add_run_output_arguments(recover)
    return parser


def _add_run_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", default=None, help="可选 dotenv 文件路径")
    parser.add_argument("--json", action="store_true", help="输出稳定紧凑 JSON")


def _dispatch_run(args: argparse.Namespace) -> int:
    config_path = Path(args.agent_config)
    env_file = Path(args.env_file) if args.env_file else None
    if args.action == "start":
        return run_start(
            RunStartOptions(
                config_path=config_path,
                input=args.input,
                session_id=args.session_id,
                run_id=args.run_id,
                max_steps=args.max_steps,
                include_tools=not args.no_tools,
                env_file=env_file,
                json_output=args.json,
            )
        )
    if args.action == "status":
        return run_status(
            RunStatusOptions(
                config_path=config_path,
                run_id=args.run_id,
                env_file=env_file,
                json_output=args.json,
            )
        )
    if args.action == "events":
        return run_events(
            RunEventsOptions(
                config_path=config_path,
                run_id=args.run_id,
                after_sequence=args.after_sequence,
                env_file=env_file,
                json_output=args.json,
            )
        )
    if args.action == "resume":
        return run_resume(
            RunResumeOptions(
                config_path=config_path,
                run_id=args.run_id,
                interaction_id=args.interaction_id,
                decision=args.decision,
                answer=args.answer,
                env_file=env_file,
                json_output=args.json,
            )
        )
    if args.action == "cancel":
        return run_cancel(
            RunCancelOptions(
                config_path=config_path,
                run_id=args.run_id,
                reason=args.reason,
                settlement_timeout=args.settlement_timeout,
                env_file=env_file,
                json_output=args.json,
            )
        )
    if args.action == "recover":
        return run_recover(
            RunRecoverOptions(
                config_path=config_path,
                run_id=args.run_id,
                activation_id=args.activation_id,
                env_file=env_file,
                json_output=args.json,
            )
        )
    raise ValueError(f"未知 run action: {args.action}")


__all__ = ["main"]
