"""离线汇总一个 Harbor job；原始运行、评分与归档文件始终只读。"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal

_METRICS = (
    "input_tokens",
    "output_tokens",
    "compaction_input_tokens",
    "compaction_output_tokens",
    "model_steps",
    "tool_calls",
    "cache_tokens",
    "cost_usd",
    "total_seconds",
    "environment_setup_seconds",
    "agent_setup_seconds",
    "agent_seconds",
    "verifier_seconds",
)


@dataclass(frozen=True, slots=True)
class _ReviewDecision:
    """从显式复核文件解析得到的评分有效性决定。"""

    reviewed_score_status: Literal["valid", "invalid", "missing"]
    raw_reward: float | None
    category: str
    reason: str
    retry_as_infrastructure: bool


def _read_review_decisions(path: Path) -> dict[str, _ReviewDecision]:
    """在文件输入边界解析复核字段；原始 reward 只作为关联依据。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: 复核文件必须是 trial 名到决定的 JSON object")
    required = {
        "reviewed_score_status",
        "raw_reward",
        "category",
        "reason",
        "retry_as_infrastructure",
    }
    decisions = {}
    for trial_name, value in data.items():
        prefix = f"{path}: {trial_name}"
        if not trial_name.strip() or not isinstance(value, dict):
            raise ValueError(f"{prefix}: trial 名必须非空，复核决定必须是 JSON object")
        missing = required - value.keys()
        if missing:
            raise ValueError(f"{prefix}: 缺少复核字段 {', '.join(sorted(missing))}")
        status = value["reviewed_score_status"]
        if status not in ("valid", "invalid", "missing"):
            raise ValueError(f"{prefix}: reviewed_score_status 必须为 valid、invalid 或 missing")
        reward = value["raw_reward"]
        if reward is not None and type(reward) not in (int, float):
            raise ValueError(f"{prefix}: raw_reward 必须是数值或 null")
        if (status == "missing") != (reward is None):
            raise ValueError(f"{prefix}: {status} 与 raw_reward 不一致，只有 missing 使用 null")
        if status == "valid" and reward not in (0, 1):
            raise ValueError(f"{prefix}: valid 必须对应原始二值 raw_reward")
        for field in ("category", "reason"):
            if not isinstance(value[field], str) or not value[field].strip():
                raise ValueError(f"{prefix}: {field} 必须是非空字符串")
        if not isinstance(value["retry_as_infrastructure"], bool):
            raise ValueError(f"{prefix}: retry_as_infrastructure 必须是布尔值")
        decisions[trial_name] = _ReviewDecision(
            reviewed_score_status=status,
            raw_reward=reward,
            category=value["category"],
            reason=value["reason"],
            retry_as_infrastructure=value["retry_as_infrastructure"],
        )
    return decisions


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _seconds(timing: dict[str, Any] | None) -> float | None:
    timing = timing or {}
    start, finish = timing.get("started_at"), timing.get("finished_at")
    if start is None or finish is None:
        return None
    return (datetime.fromisoformat(finish) - datetime.fromisoformat(start)).total_seconds()


def _empty_row(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "occurrence": None,
        "trial_name": None,
        "trial_path": None,
        "started_at": None,
        "finished_at": None,
        "execution_status": "not_started",
        "execution_error": None,
        "exception_type": None,
        "exception_message": None,
        "reward": None,
        "rewards": None,
        "automatic_score_status": "missing",
        "reviewed_score_status": None,
        "score_status": "missing",
        "review_category": None,
        "review_reason": None,
        "retry_as_infrastructure": None,
        "cleanup_completed": None,
        "cleanup_error": None,
        "archive_status": "missing",
        "missing_archive_files": [],
        "review_materials_status": "missing",
        "missing_review_files": [],
        "task_artifacts_status": "unknown",
        "artifact_entries": [],
        "missing_artifact_files": [],
        "model": None,
        **dict.fromkeys(_METRICS),
    }


def _review_materials(path: Path, row: dict[str, Any], result: dict[str, Any]) -> None:
    """分开记录评分复查材料和 Harbor best-effort 任务产物采集。"""
    missing = [] if (path / "result.json").is_file() else ["result.json"]
    verifier_ran = row["reward"] is not None or result.get("verifier") is not None
    verifier_error = row["exception_type"] or ""
    if verifier_ran or "Verifier" in verifier_error or "RewardFile" in verifier_error:
        if not (path / "verifier/test-stdout.txt").is_file():
            missing.append("verifier/test-stdout.txt")
    if row["reward"] is not None and not any(
        (path / "verifier" / name).is_file() for name in ("reward.txt", "reward.json")
    ):
        missing.extend(("verifier/reward.txt", "verifier/reward.json"))
    row["missing_review_files"] = missing
    row["review_materials_status"] = (
        "complete" if row["archive_status"] == "complete" and not missing else "partial"
    )
    manifest = path / "artifacts/manifest.json"
    if not manifest.is_file():
        return
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    row["artifact_entries"] = entries
    missing_artifacts = [
        entry["destination"]
        for entry in entries
        if entry["status"] == "ok" and not (path / entry["destination"]).exists()
    ]
    row["missing_artifact_files"] = missing_artifacts
    if missing_artifacts:
        row["task_artifacts_status"] = "partial"
    elif entries and all(entry["status"] in ("ok", "empty") for entry in entries):
        row["task_artifacts_status"] = (
            "complete" if any(entry["status"] == "ok" for entry in entries) else "unknown"
        )
    # failed 未区分源不存在与下载失败，skipped 也没有采集成功证据，维持 unknown。


def _trial_row(path: Path) -> dict[str, Any]:
    result = _read(path / "result.json")
    config = result.get("config") or _read(path / "config.json")
    task_id = result.get("task_name")
    if task_id is None:
        task_id = config["task"].get("name") or (
            config["task"]["path"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        )
    row = _empty_row(task_id)
    iris_dir = path / "agent/iris"
    iris = _read(iris_dir / "result.json")
    trace = _read(iris_dir / "trace.json")
    exception = result.get("exception_info") or {}
    usage = iris.get("usage")
    agent = result.get("agent_result") or {}
    execution_error = iris.get("execution_error")
    sdk_error = (iris.get("result") or {}).get("error")
    row.update(
        trial_name=result.get("trial_name", path.name),
        trial_path=str(path.resolve()),
        started_at=result.get("started_at"),
        finished_at=result.get("finished_at"),
        execution_status=(
            "execution_error"
            if execution_error or sdk_error
            else iris.get("stop_reason")
            or iris.get("phase")
            or ("harbor_error" if exception else "incomplete")
        ),
        execution_error=execution_error or sdk_error,
        exception_type=exception.get("exception_type"),
        exception_message=exception.get("exception_message"),
        cleanup_completed=iris.get("cleanup_completed"),
        cleanup_error=iris.get("cleanup_error"),
        model=iris.get("model") or (config.get("agent") or {}).get("model_name"),
        cost_usd=agent.get("cost_usd"),
        cache_tokens=agent.get("n_cache_tokens"),
        total_seconds=_seconds(result),
        environment_setup_seconds=_seconds(result.get("environment_setup")),
        agent_setup_seconds=_seconds(result.get("agent_setup")),
        agent_seconds=_seconds(result.get("agent_execution")),
        verifier_seconds=_seconds(result.get("verifier")),
    )
    if usage is not None:
        compaction = usage["compaction"]
        row.update(
            input_tokens=usage["input_tokens"] + compaction["input_tokens"],
            output_tokens=usage["output_tokens"] + compaction["output_tokens"],
            compaction_input_tokens=compaction["input_tokens"],
            compaction_output_tokens=compaction["output_tokens"],
            model_steps=usage["model_steps_committed"],
            tool_calls=usage["tool_calls_committed"],
        )
    rewards = (result.get("verifier_result") or {}).get("rewards")
    reward = None if rewards is None else rewards.get("reward")
    row.update(rewards=rewards, reward=reward)
    if reward is not None:
        # 超时和普通非零退出后 Harbor 仍可合法评分；其他异常需要复核原始日志。
        problem = (
            execution_error
            or sdk_error
            or iris.get("cleanup_completed") is False
            or exception.get("exception_type")
            not in (
                None,
                "AgentTimeoutError",
                "NonZeroAgentExitCodeError",
            )
        )
        row["score_status"] = "valid" if reward in (0, 1) and not problem else "review_required"
    if iris:
        row["archive_status"] = iris["trace_status"]
        missing = [
            name for name in ("trace.json", "lifecycle.db") if not (iris_dir / name).is_file()
        ]
        for asset in trace.get("assets", []):
            if asset.get("path") is not None and not (iris_dir / asset["path"]).is_file():
                missing.append(asset["path"])
        row["missing_archive_files"] = missing
        if row["archive_status"] == "complete":
            row["archive_status"] = (
                "partial" if missing else trace.get("export", {}).get("status", "missing")
            )
    _review_materials(path, row, result)
    row["automatic_score_status"] = row["score_status"]
    return row


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_job(
    job_dir: Path,
    *,
    task_ids: list[str],
    attempts: int,
    review_decisions_path: Path | None = None,
) -> dict[str, Any]:
    """按显式任务清单和次数汇总，额外 trial 单列，不从已完成数量推断分母。

    Args:
        job_dir: 单个 Harbor job 目录，不接受跨 job 的隐式合并。
        task_ids: 本批计划 task ID，校准批次只传校准任务。
        attempts: 每题计划执行次数。
        review_decisions_path: 可选的显式复核文件，只调整评分有效性，不覆盖原始分数。

    Returns:
        dict[str, Any]: 逐 trial、按题统计与覆盖率；未知用量保留空值。

    Raises:
        ValueError: 复核字段无效，或复核引用的 raw_reward 与实际 trial 不一致。
    """
    decisions = (
        _read_review_decisions(review_decisions_path) if review_decisions_path is not None else {}
    )
    unused_reviews = set(decisions)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(job_dir.iterdir()):
        if path.is_dir() and ((path / "result.json").is_file() or (path / "config.json").is_file()):
            row = _trial_row(path)
            if decision := decisions.get(row["trial_name"]):
                if decision.raw_reward != row["reward"]:
                    raise ValueError(
                        f"{row['trial_name']}: 复核 raw_reward 与实际原始 reward 不一致"
                    )
                row.update(
                    reviewed_score_status=decision.reviewed_score_status,
                    score_status=decision.reviewed_score_status,
                    review_category=decision.category,
                    review_reason=decision.reason,
                    retry_as_infrastructure=decision.retry_as_infrastructure,
                )
                unused_reviews.remove(row["trial_name"])
            grouped[row["task_id"]].append(row)
    rows: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    by_task: dict[str, Any] = {}
    for task_id in task_ids:
        trials = sorted(
            grouped.pop(task_id, []), key=lambda r: (r["started_at"] or "", r["trial_name"])
        )
        for index, row in enumerate(trials, start=1):
            row["occurrence"] = index
        selected = trials[:attempts]
        for index in range(len(selected), attempts):
            row = _empty_row(task_id)
            row["occurrence"] = index + 1
            selected.append(row)
        rows.extend(selected)
        extra.extend(trials[attempts:])
        valid = [r for r in selected if r["score_status"] == "valid"]
        by_task[task_id] = {
            "planned": attempts,
            "valid_scored": len(valid),
            "passed": sum(r["reward"] == 1 for r in valid),
        }
    extra.extend(row for group in grouped.values() for row in group)
    valid = [row for row in rows if row["score_status"] == "valid"]
    started = [row for row in rows if row["trial_name"] is not None]
    passed = sum(row["reward"] == 1 for row in valid)
    complete = sum(row["archive_status"] == "complete" for row in started)
    review_complete = sum(row["review_materials_status"] == "complete" for row in started)
    metrics: dict[str, Any] = {}
    for name in _METRICS:
        values = sorted(row[name] for row in rows if row[name] is not None)
        metric = {"known_sum": sum(values) if values else None, "known_trials": len(values)}
        if name.endswith("_seconds"):
            metric.update(
                median=median(values) if values else None,
                p90=values[math.ceil(len(values) * 0.9) - 1] if values else None,
            )
        metrics[name] = metric
    config = _read(job_dir / "config.json")
    job_result = _read(job_dir / "result.json")
    return {
        "job_name": config.get("job_name", job_dir.name),
        "job_path": str(job_dir.resolve()),
        "job_started_at": job_result.get("started_at"),
        "job_finished_at": job_result.get("finished_at"),
        "configured_attempts": config.get("n_attempts"),
        "configured_concurrency": config.get("n_concurrent_trials"),
        "configured_retry": config.get("retry"),
        "task_ids": task_ids,
        "attempts": attempts,
        "unused_review_trials": sorted(unused_reviews),
        "counts": {
            "planned": len(rows),
            "started": len(started),
            "not_started": len(rows) - len(started),
            "valid_scored": len(valid),
            "passed": passed,
            "exceptions": sum(row["exception_type"] is not None for row in rows),
            "archive_complete": complete,
            "review_materials_complete": review_complete,
            "extra_trials": len(extra),
        },
        "rates": {
            "planned_success": _ratio(passed, len(rows)),
            "scored_success": _ratio(passed, len(valid)),
            "score_coverage": _ratio(len(valid), len(rows)),
            "archive_coverage": _ratio(complete, len(started)),
            "review_materials_coverage": _ratio(review_complete, len(started)),
        },
        "pass_at_1": _ratio(passed, len(rows)) if len(valid) == len(rows) else None,
        "by_task": by_task,
        "metrics": metrics,
        "trials": rows,
        "extra_trials": extra,
    }


def _markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    n, v, p = (counts[key] for key in ("planned", "valid_scored", "passed"))
    lines = [
        f"# {report['job_name']} 离线汇总",
        "",
        f"计划成功：{p} / {n}；有效评分成功：{p} / {v}；有效评分覆盖：{v} / {n}。",
        f"已启动 {counts['started']}，未启动 {counts['not_started']}，异常 {counts['exceptions']}，"
        f"额外 trial {counts['extra_trials']}。",
        f"轨迹与评分材料可复查：{counts['review_materials_complete']} / {counts['started']}；"
        f"Iris 归档完整：{counts['archive_complete']} / {counts['started']}。",
        "",
        "| task | 通过 / 计划 | 有效评分 |",
        "| --- | ---: | ---: |",
    ]
    for task, values in report["by_task"].items():
        lines.append(
            f"| {task} | {values['passed']} / {values['planned']} | {values['valid_scored']} |"
        )
    lines.extend(
        [
            "",
            "## 逐 trial",
            "",
            "| 范围 | task / 序号 | 执行 | reward | 评分 | 收尾 | 归档 | 可复查 | 产物 | 材料 |",
            "| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for scope, rows in (("计划", report["trials"]), ("额外", report["extra_trials"])):
        for row in rows:
            path = row["trial_path"]
            link = f"[trial](<{path}/>) · [verifier](<{path}/verifier/>)" if path else ""
            lines.append(
                f"| {scope} | {row['task_id']} / {row['occurrence']} | {row['execution_status']} | "
                f"{'' if row['reward'] is None else row['reward']} | {row['score_status']} | "
                f"{row['cleanup_completed']} | {row['archive_status']} | "
                f"{row['review_materials_status']} | {row['task_artifacts_status']} | {link} |"
            )
    reviewed = [
        row
        for row in report["trials"] + report["extra_trials"]
        if row["reviewed_score_status"] is not None
    ]
    if reviewed:
        lines.extend(
            [
                "",
                "## 显式复核",
                "",
                "| trial | 自动评分 | 复核评分 | 分类 | 理由 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in reviewed:
            values = (
                row["trial_name"],
                row["automatic_score_status"],
                row["reviewed_score_status"],
                row["review_category"],
                row["review_reason"],
            )
            cells = [value.replace("|", r"\|").replace("\n", "<br>") for value in values]
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## 用量与耗时",
            "",
            "| 字段 | 已知合计 | 覆盖 trial | 中位数 / P90 |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for name, values in report["metrics"].items():
        total = values["known_sum"]
        percentiles = f"{values['median']} / {values['p90']}" if "median" in values else ""
        lines.append(
            f"| {name} | {'' if total is None else total} | "
            f"{values['known_trials']} | {percentiles} |"
        )
    lines.extend(
        [
            "",
            "未知费用和用量保持空值；已知合计仅覆盖对应数量的 trial。",
            "occurrence 是任务内按开始时间排序的展示序号，不代表受控 seed。",
            "额外 trial 不进入计划统计。",
            "review_required 保留原始 reward，需复核错误和 verifier 日志；不计入有效评分。",
            "提供显式复核时采用 reviewed_score_status；否则采用 automatic_score_status。",
            "复核只改变评分有效性，raw_reward 用于核对，不替换原始 reward；重跑标记不触发执行。",
            "复核文件中未匹配本 job 的 trial 名列于 summary.json 的 unused_review_trials。",
            "Iris 收尾字段来自 SDK 回执；归档字段检查 Iris trace、数据库和引用附件。",
            "可复查覆盖在 Iris 归档基础上检查 verifier 日志和已有评分的原始 reward 文件。",
            "任务产物单列：manifest 的 failed 无法区分不存在源和下载异常，按 unknown 展示。",
            "Harbor 没有单列收尾耗时，报告不把 verifier 完成后的日志下载时间冒充 cleanup 时间。",
            "P90 使用 nearest-rank；详细错误、原始 rewards 和文件缺失清单见 summary.json。",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    """输出 CSV、JSON 和 Markdown，将材料路径写为相对报告目录的索引。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = deepcopy(report)
    saved["job_path"] = Path(os.path.relpath(saved["job_path"], output_dir)).as_posix()
    for row in saved["trials"] + saved["extra_trials"]:
        if row["trial_path"] is not None:
            row["trial_path"] = Path(os.path.relpath(row["trial_path"], output_dir)).as_posix()
    (output_dir / "summary.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output_dir / "trials.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["planned", *_empty_row("")])
        writer.writeheader()
        for planned, rows in ((True, saved["trials"]), (False, saved["extra_trials"])):
            for row in rows:
                writer.writerow(
                    {
                        "planned": planned,
                        **{
                            key: json.dumps(value, ensure_ascii=False)
                            if isinstance(value, (dict, list))
                            else value
                            for key, value in row.items()
                        },
                    }
                )
    (output_dir / "report.md").write_text(_markdown(saved), encoding="utf-8")
