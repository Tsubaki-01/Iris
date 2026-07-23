"""DeepSeek live 验证报告输出。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .catalog import SCENARIO_CATALOG
from .config import mask_secret
from .constants import API_KEY_ENV_VAR, DEFAULT_PROVIDER_ROUTE
from .models import ScenarioReport


def scenario_report(
    *,
    name: str,
    ok: bool,
    status: str,
    api_calls: int,
    steps: int,
    expected: str,
    actual: str,
    evidence: Mapping[str, Any],
    error_code: str = "",
    error_message: str = "",
) -> ScenarioReport:
    """构造稳定、JSON-safe 的场景报告。"""
    return {
        "name": name,
        "ok": ok,
        "status": status,
        "api_calls": api_calls,
        "steps": steps,
        "expected": expected,
        "actual": actual,
        "evidence": dict(evidence),
        "error_code": error_code,
        "error_message": error_message,
    }


def aggregate_report(
    work_dir: Path,
    scenarios: list[ScenarioReport],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """聚合所有场景报告。"""
    for scenario in scenarios:
        _attach_catalog_fields(scenario)
    failed = [scenario["name"] for scenario in scenarios if not scenario["ok"]]
    module_coverage = _module_coverage(scenarios)
    failure_summary = [
        {
            "name": scenario["name"],
            "status": scenario["status"],
            "error_code": scenario["error_code"],
            "error_message": scenario["error_message"],
            "actual": scenario["actual"],
        }
        for scenario in scenarios
        if not scenario["ok"]
    ]
    return {
        "schema_version": 1,
        "ok": not failed,
        "work_dir": str(work_dir),
        "metadata": dict(metadata or {}),
        "scenario_catalog": SCENARIO_CATALOG,
        "scenario_count": len(scenarios),
        "total_api_calls": sum(int(scenario["api_calls"]) for scenario in scenarios),
        "total_steps": sum(int(scenario["steps"]) for scenario in scenarios),
        "failed_scenarios": failed,
        "blocking_scenarios": failed,
        "blocking_modules": [module["module"] for module in module_coverage if not module["ok"]],
        "module_coverage": module_coverage,
        "failure_summary": failure_summary,
        "scenarios": scenarios,
    }


def write_report(work_dir: Path, report: dict[str, Any]) -> Path:
    """把聚合报告写入 work dir，并在报告中记录路径。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / "report.json"
    summary_path = work_dir / "summary.md"
    report["report_path"] = str(report_path)
    report["summary_path"] = str(summary_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(_render_summary(report), encoding="utf-8")
    return report_path


def _render_summary(report: dict[str, Any]) -> str:
    """渲染给人工快速检查的 Markdown 摘要。"""
    lines = [
        "# DeepSeek Flow Summary",
        "",
        "## Result",
        "",
        f"- Result: {'PASS' if report['ok'] else 'FAIL'}",
        f"- Scenarios: {report['scenario_count']}",
        f"- Total API calls: {report['total_api_calls']}",
        f"- Total steps: {report['total_steps']}",
        f"- Work dir: {report['work_dir']}",
        f"- Report JSON: {report.get('report_path', '')}",
    ]
    metadata = report.get("metadata", {})
    if metadata:
        lines.extend(
            [
                f"- Log dir: {metadata.get('log_dir', '')}",
                f"- Provider route: {metadata.get('provider_route', '')}",
                f"- Scenario selector: {metadata.get('scenario', '')}",
                f"- Retries: {metadata.get('retries', '')}",
                f"- Duration seconds: {metadata.get('duration_seconds', '')}",
            ]
        )
        environment = metadata.get("environment", {})
        git = environment.get("git", {}) if isinstance(environment, dict) else {}
        if isinstance(git, dict) and git:
            lines.extend(
                [
                    f"- Git commit: {git.get('commit', '')}",
                    f"- Git branch: {git.get('branch', '')}",
                    f"- Git dirty: {git.get('dirty', '')}",
                ]
            )

    lines.extend(["", "## Blocking Scenarios", ""])
    if report["blocking_scenarios"]:
        for scenario_name in report["blocking_scenarios"]:
            lines.append(f"- {scenario_name}")
    else:
        lines.append("- none")

    lines.extend(["", "## Module Coverage", ""])
    lines.append(
        "| Module | Result | Scenarios | API calls | Steps | Runtime APIs | Failed scenarios |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- |")
    for module in report["module_coverage"]:
        result = "PASS" if module["ok"] else "FAIL"
        runtime_apis = ", ".join(module["runtime_apis"])
        failed_scenarios = ", ".join(module["failed_scenarios"])
        lines.append(
            f"| {module['module']} | {result} | {module['scenario_count']} | "
            f"{module['api_calls']} | {module['steps']} | {runtime_apis} | "
            f"{failed_scenarios} |"
        )

    lines.extend(["", "## Failure Summary", ""])
    if report["failure_summary"]:
        for failure in report["failure_summary"]:
            scenario = _scenario_by_name(report, failure["name"])
            lines.extend(
                [
                    f"### {failure['name']}",
                    "",
                    f"- Status: {failure['status']}",
                    f"- Error code: {failure['error_code']}",
                    f"- Error message: {failure['error_message']}",
                    f"- Actual: {failure['actual']}",
                    f"- Scenario dir: {scenario.get('scenario_dir', '')}",
                    "",
                ]
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Scenario Matrix", ""])
    lines.append(
        "| Scenario | Module | Runtime API | Result | Status | "
        "API calls | Steps | Scenario dir | Error |"
    )
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |")
    for scenario in report["scenarios"]:
        result = "PASS" if scenario["ok"] else "FAIL"
        error = scenario["error_code"] or scenario["error_message"]
        lines.append(
            f"| {scenario['name']} | {scenario.get('module', '')} | "
            f"{scenario.get('runtime_api', '')} | {result} | {scenario['status']} | "
            f"{scenario['api_calls']} | {scenario['steps']} | "
            f"{scenario.get('scenario_dir', '')} | {error} |"
        )
    lines.append("")
    return "\n".join(lines)


def _scenario_by_name(report: dict[str, Any], name: str) -> dict[str, Any]:
    """按名称读取场景报告。"""
    for scenario in report["scenarios"]:
        if scenario["name"] == name:
            return scenario
    return {}


def _attach_catalog_fields(scenario: ScenarioReport) -> None:
    """把场景目录字段写入报告。"""
    entry = SCENARIO_CATALOG.get(str(scenario["name"]), {})
    scenario.setdefault("module", entry.get("module", ""))
    scenario.setdefault("runtime_api", entry.get("runtime_api", ""))
    scenario.setdefault("uses_deepseek", entry.get("uses_deepseek", True))
    scenario.setdefault("description", entry.get("description", ""))


def _module_coverage(scenarios: list[ScenarioReport]) -> list[dict[str, Any]]:
    """按业务模块聚合场景报告。"""
    modules: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        module_name = str(scenario.get("module", ""))
        module = modules.setdefault(
            module_name,
            {
                "module": module_name,
                "ok": True,
                "scenario_count": 0,
                "failed_scenarios": [],
                "api_calls": 0,
                "steps": 0,
                "runtime_apis": [],
            },
        )
        module["scenario_count"] += 1
        module["api_calls"] += int(scenario["api_calls"])
        module["steps"] += int(scenario["steps"])
        runtime_api = str(scenario.get("runtime_api", ""))
        if runtime_api and runtime_api not in module["runtime_apis"]:
            module["runtime_apis"].append(runtime_api)
        if not scenario["ok"]:
            module["ok"] = False
            module["failed_scenarios"].append(scenario["name"])
    return list(modules.values())


def print_intro(console: Console, *, api_key: str, work_dir: Path, log_dir: Path) -> None:
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
    table.add_column("场景", style="bold")
    table.add_column("状态")
    table.add_column("API")
    table.add_column("关键输出", overflow="fold")

    for scenario in report["scenarios"]:
        table.add_row(
            scenario["name"],
            _status_label(bool(scenario["ok"])),
            str(scenario["api_calls"]),
            (
                f"status={scenario['status']}; steps={scenario['steps']}; "
                f"expected={scenario['expected']}; actual={scenario['actual']}; "
                f"error={scenario['error_code'] or scenario['error_message']}"
            ),
        )
    console.print(table)

    summary_style = "green" if report["ok"] else "red"
    summary = "全流程验证通过" if report["ok"] else "全流程验证未通过"
    if report.get("report_path"):
        summary = f"{summary}\nreport: {report['report_path']}"
    console.print(Panel(summary, title="结论", border_style=summary_style))


def _status_label(ok: bool) -> str:
    """返回 Rich 状态文本。"""
    return "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
