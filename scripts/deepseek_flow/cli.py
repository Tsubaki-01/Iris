"""DeepSeek live 验证命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from tempfile import mkdtemp

from rich.console import Console
from rich.panel import Panel

from iris.log import logger

from .bootstrap import ROOT
from .config import init_local_config, resolve_api_key, setup_flow_logging
from .constants import API_KEY_ENV_VAR, SCENARIO_NAMES
from .reporting import print_intro, print_report, write_report
from .runner import run_deepseek_flow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="验证 Iris 当前 DeepSeek provider 与 runtime 全流程。",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="保留验证文件的运行目录；不传则使用临时目录。",
    )
    parser.add_argument(
        "--scenario",
        choices=("all", *SCENARIO_NAMES),
        default="all",
        help="只运行指定 live 场景；默认运行全部场景。",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="模型未按要求调用工具时的重试次数。",
    )
    return parser.parse_args(argv)


async def amain(argv: list[str] | None = None) -> int:
    """异步主入口。"""
    _configure_output_encoding()
    args = parse_args(argv)
    console = Console()
    init_local_config(ROOT)
    api_key = resolve_api_key()
    if api_key is None:
        console.print(
            Panel(
                (
                    f"缺少 {API_KEY_ENV_VAR}。请写入 .env.local，或在当前 shell "
                    f"设置 {API_KEY_ENV_VAR} 后重试。"
                ),
                title="配置缺失",
                border_style="red",
            )
        )
        return 2

    if args.retries < 0:
        console.print(Panel("--retries 必须非负。", title="参数错误", border_style="red"))
        return 2

    if args.work_dir is not None:
        work_dir = args.work_dir
    else:
        work_dir = Path(mkdtemp(prefix="iris-deepseek-flow-"))

    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir = work_dir / "logs"
    setup_flow_logging(log_dir)
    print_intro(console, api_key=api_key, work_dir=work_dir, log_dir=log_dir)
    report = await run_deepseek_flow(
        work_dir=work_dir,
        scenario=args.scenario,
        retries=args.retries,
        log_dir=log_dir,
    )
    report_path = write_report(work_dir, report)
    logger.info("deepseek.report.written path={}", report_path)

    print_report(console, report)
    return 0 if report["ok"] else 1


def main() -> None:
    """同步脚本入口。"""
    raise SystemExit(asyncio.run(amain()))


def _configure_output_encoding() -> None:
    """在 Windows 终端捕获场景下优先输出 UTF-8。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
