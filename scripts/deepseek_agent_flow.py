"""使用 DeepSeek 验证 Iris 当前 provider 与 runtime 全流程。

脚本会通过 Iris 集中配置读取本地 `.env.local` / `.env` 中的
`IRIS_PROVIDER_API_KEYS__DEEPSEEK` 或 `IRIS_API_KEY`，创建临时 `agent.yaml` 和
workspace 文件，然后依次验证：

1. provider factory 是否能创建 DeepSeek client 并完成一次直接模型调用。
2. runtime 是否能从 YAML 装配 Agent、挂载 `file.read` 工具、完成有界 tool loop。

Example:
    uv run python scripts/deepseek_agent_flow.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from iris.config import get_config, init_config, is_config_initialized  # noqa: E402
from iris.exceptions import IrisConfigError  # noqa: E402
from iris.log import logger, setup_logger  # noqa: E402
from iris.message import LLMRequest, Msg  # noqa: E402
from iris.providers import create_provider_client  # noqa: E402
from iris.runtime import RuntimeProvider  # noqa: E402
from iris.runtime.factory import RuntimeFactory  # noqa: E402
from iris.runtime.models import BoundedLoopOptions, RuntimeOptions  # noqa: E402

API_KEY_ENV_VAR = "IRIS_PROVIDER_API_KEYS__DEEPSEEK / IRIS_API_KEY"
LOCAL_ENV_FILES = (".env.local", ".env")
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_PROVIDER_ROUTE = f"deepseek/{DEFAULT_MODEL}"


def init_local_config(base_dir: Path = ROOT) -> bool:
    """通过 Iris 集中配置初始化脚本运行配置。

    Args:
        base_dir: 仓库根目录或测试用目录。

    Returns:
        配置初始化成功则为 True，缺少必需配置则为 False。
    """
    if is_config_initialized():
        return True

    for env_path in _local_env_paths(base_dir):
        if _try_init_config(env_path):
            return True
    return _try_init_config(None)


def resolve_api_key() -> str | None:
    """解析当前配置中 DeepSeek 可用的 API key。"""
    if not is_config_initialized():
        return None
    config = get_config()
    return config.provider_api_keys.get("deepseek") or config.api_key


def _local_env_paths(base_dir: Path) -> list[Path]:
    """返回按优先级存在的本地 env 文件路径。"""
    return [
        base_dir / file_name
        for file_name in LOCAL_ENV_FILES
        if (base_dir / file_name).exists()
    ]


def _try_init_config(env_path: Path | None) -> bool:
    """尝试用指定 env 文件初始化 Iris 配置。"""
    try:
        if env_path is None:
            init_config()
        else:
            init_config(env_file=str(env_path))
    except (IrisConfigError, ValidationError):
        return False
    return True


def mask_secret(value: str | None) -> str:
    """返回适合打印的 secret 掩码。"""
    if not value:
        return "<missing>"
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:3]}...{value[-4:]}"


def setup_flow_logging(log_dir: Path) -> None:
    """配置 DeepSeek 验证脚本的文件日志。

    Args:
        log_dir: 日志输出目录。
    """
    setup_logger(log_dir)
    logger.info("deepseek.logging.configured log_dir={}", log_dir)


def prepare_runtime_workspace(base_dir: Path) -> tuple[Path, str]:
    """准备 runtime 验证用的 agent 配置与只读 workspace。

    Args:
        base_dir: 临时运行目录。

    Returns:
        `agent.yaml` 路径和需要模型回读的验证码。
    """
    workspace = base_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    expected_token = "deepseek-flow-token-2026-07-04"
    (workspace / "verification.txt").write_text(
        f"验证码: {expected_token}\n",
        encoding="utf-8",
    )

    agent_path = base_dir / "agent.yaml"
    agent_path.write_text(
        f"""
name: deepseek-flow-check
model:
  provider: deepseek
  name: {DEFAULT_MODEL}
  temperature: 0
  max_tokens: 200
  timeout: 60
system: |
  你是 Iris DeepSeek 全流程验证助手。
  当用户要求读取文件时，必须先调用 read_file 工具读取文件内容。
  最终回答保持一行，并原样包含读取到的验证码。
tools:
  builtin:
    - file.read
permissions:
  workspace: workspace
  writes: deny
session:
  backend: none
""".strip(),
        encoding="utf-8",
    )
    logger.info(
        "deepseek.workspace.prepared agent_path={} workspace={} expected_token={}",
        agent_path,
        workspace,
        expected_token,
    )
    return agent_path, expected_token


async def run_provider_smoke(
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """验证 provider factory 与 DeepSeek 直接调用。

    Args:
        model: DeepSeek 模型名。

    Returns:
        JSON-safe 诊断信息。
    """
    logger.info("deepseek.provider.start route=deepseek/{}", model)
    client = create_provider_client(f"deepseek/{model}", timeout=60)
    try:
        response = await client.complete(
            LLMRequest(
                model=model,
                messages=[
                    Msg.user(
                        "这是连通性验证。请只回答 IRIS_PROVIDER_OK，不要添加其他内容。"
                    )
                ],
                temperature=0,
                max_tokens=32,
                timeout=60,
            )
        )
    except Exception:
        logger.exception("deepseek.provider.error route=deepseek/{}", model)
        raise

    text = response.to_msg().text.strip()
    logger.info(
        "deepseek.provider.finish provider={} model={} total_tokens={} output={}",
        response.provider,
        response.model or model,
        response.total_tokens,
        text,
    )
    return {
        "ok": bool(text),
        "provider": response.provider,
        "model": response.model or model,
        "text": text,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
    }


async def run_runtime_tool_loop(
    *,
    agent_path: Path,
    expected_token: str,
    provider: RuntimeProvider | None = None,
) -> dict[str, Any]:
    """验证 YAML -> runtime -> provider -> tool bridge -> final answer 流程。

    Args:
        agent_path: 验证用 `agent.yaml`。
        expected_token: 需要从文件工具读回的验证码。
        provider: 测试时可注入的 provider；不传则创建真实 DeepSeek client。

    Returns:
        JSON-safe 诊断信息。
    """
    logger.info(
        "deepseek.runtime.start agent_path={} expected_token={} injected_provider={}",
        agent_path,
        expected_token,
        provider is not None,
    )
    runtime = RuntimeFactory.from_config_path(
        agent_path,
        provider=provider,
    )
    result = await runtime.run_loop(
        (
            "请使用 read_file 工具读取 verification.txt，然后只用一行回答："
            "IRIS_RUNTIME_TOOL_OK: <验证码>"
        ),
        options=RuntimeOptions(
            session_id="deepseek-flow-check",
            loop=BoundedLoopOptions(max_steps=4),
            metadata={"script": "deepseek_agent_flow"},
        ),
    )

    final_text = (
        result.assistant_message.text.strip() if result.assistant_message else ""
    )
    expected_token_found = expected_token in final_text
    tool_result_count = len(result.tool_results)
    logger.info(
        (
            "deepseek.runtime.finish status={} steps={} provider_request_count={} "
            "tool_result_count={} expected_token_found={} final_text={} error_code={}"
        ),
        result.status.value,
        result.steps,
        _provider_request_count(provider),
        tool_result_count,
        expected_token_found,
        final_text,
        result.error.code if result.error else "",
    )
    return {
        "ok": result.status.value == "ok"
        and tool_result_count > 0
        and expected_token_found,
        "status": result.status.value,
        "steps": result.steps,
        "provider_request_count": _provider_request_count(provider),
        "tool_result_count": tool_result_count,
        "expected_token_found": expected_token_found,
        "final_text": final_text,
        "error_code": result.error.code if result.error else "",
        "error_message": result.error.message if result.error else "",
    }


async def run_deepseek_flow(
    *,
    work_dir: Path,
) -> dict[str, Any]:
    """执行完整 DeepSeek 验证流程。"""
    api_key = resolve_api_key()
    logger.info(
        "deepseek.flow.start work_dir={} route={} api_key={}",
        work_dir,
        DEFAULT_PROVIDER_ROUTE,
        mask_secret(api_key),
    )
    agent_path, expected_token = prepare_runtime_workspace(work_dir)
    provider_report = await run_provider_smoke()
    runtime_report = await run_runtime_tool_loop(
        agent_path=agent_path,
        expected_token=expected_token,
    )
    logger.info(
        "deepseek.flow.finish ok={} provider_ok={} runtime_ok={}",
        provider_report["ok"] and runtime_report["ok"],
        provider_report["ok"],
        runtime_report["ok"],
    )
    return {
        "ok": provider_report["ok"] and runtime_report["ok"],
        "work_dir": str(work_dir),
        "agent_path": str(agent_path),
        "provider": provider_report,
        "runtime": runtime_report,
    }


def print_intro(
    console: Console, *, api_key: str, work_dir: Path, log_dir: Path
) -> None:
    """打印验证输入提示。"""
    console.print(
        Panel.fit(
            "\n".join(
                [
                    "[bold]输入配置[/bold]",
                    f"provider route: [cyan]{DEFAULT_PROVIDER_ROUTE}[/cyan]",
                    f"env var: [cyan]{API_KEY_ENV_VAR}[/cyan]",
                    f"api key: [cyan]{mask_secret(api_key)}[/cyan]",
                    f"work dir: [cyan]{work_dir}[/cyan]",
                    f"log dir: [cyan]{log_dir}[/cyan]",
                ]
            ),
            title="Iris DeepSeek Flow",
            border_style="cyan",
        )
    )


def print_report(console: Console, report: dict[str, Any]) -> None:
    """打印验证输出报告。"""
    table = Table(title="验证结果", box=box.SIMPLE_HEAVY)
    table.add_column("阶段", style="bold")
    table.add_column("状态")
    table.add_column("关键输出", overflow="fold")

    provider = report["provider"]
    table.add_row(
        "Provider 直连",
        _status_label(bool(provider["ok"])),
        (
            f"model={provider['model']}; "
            f"tokens={provider['total_tokens']}; "
            f"output={provider['text']}"
        ),
    )

    runtime = report["runtime"]
    table.add_row(
        "Runtime 工具循环",
        _status_label(bool(runtime["ok"])),
        (
            f"status={runtime['status']}; steps={runtime['steps']}; "
            f"tool_results={runtime['tool_result_count']}; "
            f"token_found={runtime['expected_token_found']}; "
            f"output={runtime['final_text'] or runtime['error_message']}"
        ),
    )
    console.print(table)

    summary_style = "green" if report["ok"] else "red"
    summary = "全流程验证通过" if report["ok"] else "全流程验证未通过"
    console.print(Panel(summary, title="结论", border_style=summary_style))


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

    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        log_dir = args.work_dir / "logs"
        setup_flow_logging(log_dir)
        print_intro(console, api_key=api_key, work_dir=args.work_dir, log_dir=log_dir)
        report = await run_deepseek_flow(work_dir=args.work_dir)
    else:
        with TemporaryDirectory(prefix="iris-deepseek-flow-") as tmp_dir:
            work_dir = Path(tmp_dir)
            log_dir = work_dir / "logs"
            setup_flow_logging(log_dir)
            print_intro(console, api_key=api_key, work_dir=work_dir, log_dir=log_dir)
            report = await run_deepseek_flow(work_dir=work_dir)

    print_report(console, report)
    return 0 if report["ok"] else 1


def main() -> None:
    """同步脚本入口。"""
    raise SystemExit(asyncio.run(amain()))


def _provider_request_count(provider: RuntimeProvider | None) -> int | str:
    """从测试 provider 中读取请求次数；真实 provider 不暴露该字段。"""
    if provider is None or not hasattr(provider, "requests"):
        return "n/a"
    requests = provider.requests  # type: ignore[attr-defined]
    if isinstance(requests, list):
        return len(requests)
    return "n/a"


def _status_label(ok: bool) -> str:
    """返回 Rich 状态文本。"""
    return "[green]PASS[/green]" if ok else "[red]FAIL[/red]"


def _configure_output_encoding() -> None:
    """在 Windows 终端捕获场景下优先输出 UTF-8。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
