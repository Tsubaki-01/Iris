"""通过 Iris ToolExecutor 搜索网页、读取正文及续读长结果。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from iris.agents import ToolsConfig, build_tool_registry
from iris.config import init_config, is_config_initialized
from iris.message import ToolUseBlock
from iris.tools import ToolExecutionContext, ToolExecutor, ToolResult

WORKSPACE = Path(__file__).parent / ".iris" / "workspace"
BUILTINS = {"web_search": "web.search", "web_fetch": "web.fetch", "read_file": "file.read"}


async def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    workspace: Path,
) -> ToolResult:
    """在指定工作区执行工具，保留标准错误与 artifact 行为。

    Args:
        name: web_search、web_fetch 或 read_file。
        arguments: 工具原始参数，由 executor 完成首次校验。
        workspace: 长结果落盘与后续 read_file 使用的同一工作区。

    Returns:
        包含正文、错误或完整内容路径的标准工具结果。
    """
    registry = build_tool_registry(ToolsConfig(builtin=[BUILTINS[name]]))
    return await ToolExecutor(registry).execute_one(
        ToolUseBlock(id=uuid4().hex, name=name, input=arguments),
        ToolExecutionContext(workspace_root=workspace.resolve(), session_id="web-example"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """执行一次工具调用；工具错误返回 1，成功及部分 URL 失败返回 0。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    commands = parser.add_subparsers(dest="command", required=True)

    search = commands.add_parser("search", help="搜索来源")
    search.add_argument("query")
    search.add_argument("--max-results", type=int, default=10)
    search.add_argument("--time-range", choices=["day", "week", "month", "year"])
    search.add_argument("--include-domains", nargs="+", default=[])
    search.add_argument("--exclude-domains", nargs="+", default=[])

    fetch = commands.add_parser("fetch", help="批量读取正文或 query 摘录")
    fetch.add_argument("urls", nargs="+")
    fetch.add_argument("--query")

    read = commands.add_parser("read", help="用 read_file 续读 artifact")
    read.add_argument("file_path")
    read.add_argument("--offset", type=int, default=0)
    read.add_argument("--column", type=int, default=0)

    args = parser.parse_args(argv)
    if not is_config_initialized():
        init_config(env_file=str(args.env_file) if args.env_file is not None else None)
    arguments = vars(args).copy()
    for option in ("env_file", "workspace", "command"):
        arguments.pop(option)
    name = {"search": "web_search", "fetch": "web_fetch", "read": "read_file"}[args.command]
    result = asyncio.run(execute_tool(name, arguments, workspace=args.workspace))
    print(result.model_content)
    return int(result.is_error)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
