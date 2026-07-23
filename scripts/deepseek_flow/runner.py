"""DeepSeek live 验证场景调度。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from iris.log import logger

from .bootstrap import ROOT
from .catalog import SCENARIO_CATALOG
from .config import _safe_error_message, mask_secret, resolve_api_key
from .constants import DEFAULT_PROVIDER_ROUTE, SCENARIO_NAMES
from .environment import collect_run_environment
from .memory_scenarios import run_memory_query_live, run_memory_results_live
from .models import ScenarioReport, ScenarioRunner
from .providers import run_provider_smoke_live
from .reporting import aggregate_report, scenario_report
from .runtime_scenarios import (
    run_context_yaml_live,
    run_run_turn_live,
    run_runtime_read_loop_live,
)
from .session_scenarios import run_sqlite_session_live
from .tool_scenarios import (
    run_builtin_file_tools_live,
    run_file_not_read_recovery_live,
    run_permission_path_escape_live,
    run_python_tool_live,
    run_tool_errors_live,
)
from .utils import _scenario_dir


async def run_deepseek_flow(
    *,
    work_dir: Path,
    scenario: str = "all",
    retries: int = 2,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """执行 DeepSeek live 验证流程。"""
    started_at = datetime.now(UTC)
    started = perf_counter()
    logger.info(
        ("deepseek.flow.start work_dir={} log_dir={} route={} scenario={} retries={} api_key={}"),
        work_dir,
        log_dir,
        DEFAULT_PROVIDER_ROUTE,
        scenario,
        retries,
        mask_secret(resolve_api_key()),
    )
    reports = await run_selected_scenarios(work_dir, scenario=scenario, retries=retries)
    finished_at = datetime.now(UTC)
    report = aggregate_report(
        work_dir,
        reports,
        metadata={
            "provider_route": DEFAULT_PROVIDER_ROUTE,
            "scenario": scenario,
            "retries": retries,
            "log_dir": str(log_dir) if log_dir is not None else "",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round(perf_counter() - started, 3),
            "environment": collect_run_environment(ROOT),
        },
    )
    logger.info(
        (
            "deepseek.flow.finish ok={} scenario_count={} total_api_calls={} "
            "total_steps={} failed_scenarios={}"
        ),
        report["ok"],
        report["scenario_count"],
        report["total_api_calls"],
        report["total_steps"],
        report["failed_scenarios"],
    )
    return report


async def run_selected_scenarios(
    work_dir: Path,
    *,
    scenario: str,
    retries: int,
) -> list[ScenarioReport]:
    """运行指定场景并捕获结构化错误。"""
    names = list(SCENARIO_NAMES if scenario == "all" else (scenario,))
    reports: list[ScenarioReport] = []
    for name in names:
        logger.info("deepseek.scenario.start name={}", name)
        runner = SCENARIO_RUNNERS[name]
        scenario_dir = _scenario_dir(work_dir, name)
        try:
            report = await runner(scenario_dir, retries)
        except Exception as exc:
            error_message = _safe_error_message(exc)
            logger.error(
                "deepseek.scenario.error name={} error_code={} error_message={}",
                name,
                exc.__class__.__name__,
                error_message,
            )
            report = scenario_report(
                name=name,
                ok=False,
                status="error",
                api_calls=0,
                steps=0,
                expected="场景执行成功",
                actual=exc.__class__.__name__,
                evidence={},
                error_code=exc.__class__.__name__,
                error_message=error_message,
            )
        report["scenario_dir"] = str(scenario_dir)
        reports.append(report)
        _attach_catalog_fields(report)
        logger.info(
            "deepseek.scenario.finish name={} ok={} status={} api_calls={}",
            name,
            report["ok"],
            report["status"],
            report["api_calls"],
        )
    return reports


SCENARIO_RUNNERS: dict[str, ScenarioRunner] = {
    "provider_smoke_live": run_provider_smoke_live,
    "runtime_read_loop_live": run_runtime_read_loop_live,
    "run_turn_live": run_run_turn_live,
    "context_yaml_live": run_context_yaml_live,
    "memory_results_live": run_memory_results_live,
    "memory_query_live": run_memory_query_live,
    "sqlite_session_live": run_sqlite_session_live,
    "builtin_file_tools_live": run_builtin_file_tools_live,
    "file_not_read_recovery_live": run_file_not_read_recovery_live,
    "python_tool_live": run_python_tool_live,
    "permission_path_escape_live": run_permission_path_escape_live,
    "tool_errors_live": run_tool_errors_live,
}


def _attach_catalog_fields(report: ScenarioReport) -> None:
    """把场景目录字段写入单个报告。"""
    entry = SCENARIO_CATALOG.get(str(report["name"]), {})
    report["module"] = entry.get("module", "")
    report["runtime_api"] = entry.get("runtime_api", "")
    report["uses_deepseek"] = entry.get("uses_deepseek", True)
    report["description"] = entry.get("description", "")
