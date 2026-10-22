"""验证离线汇总保留计划分母、原始评分与独立归档状态。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from evals.reporting import summarize_job, write_report
from evals.reporting.__main__ import main


def _json(path: Path, value: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _trial(
    job: Path,
    name: str,
    task: str,
    *,
    reward: float | None,
    stop: str = "completed",
    exception: str | None = None,
    execution_error: str | None = None,
    trace_status: str = "complete",
) -> Path:
    path = job / name
    _json(
        path / "result.json",
        {
            "task_name": task,
            "trial_name": name,
            "started_at": "2026-10-22T00:00:00Z",
            "finished_at": "2026-10-22T00:00:30Z",
            "agent_execution": {
                "started_at": "2026-10-22T00:00:05Z",
                "finished_at": "2026-10-22T00:00:25Z",
            },
            "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
            "exception_info": None
            if exception is None
            else {"exception_type": exception, "exception_message": "test failure"},
            "agent_result": {"cost_usd": None, "n_cache_tokens": None},
        },
    )
    _json(
        path / "agent/iris/result.json",
        {
            "stop_reason": stop,
            "phase": "terminal",
            "execution_error": execution_error,
            "cleanup_completed": True,
            "cleanup_error": None,
            "trace_status": trace_status,
            "usage": {
                "model_steps_committed": 2,
                "tool_calls_committed": 1,
                "input_tokens": 100,
                "output_tokens": 10,
                "compaction": {"input_tokens": 20, "output_tokens": 5},
            },
        },
    )
    _json(path / "agent/iris/trace.json", {"export": {"status": trace_status}, "assets": []})
    (path / "agent/iris/lifecycle.db").touch()
    (path / "verifier").mkdir()
    (path / "verifier/test-stdout.txt").write_text("verifier output", encoding="utf-8")
    if reward is not None:
        (path / "verifier/reward.txt").write_text(str(reward), encoding="utf-8")
    _json(path / "artifacts/app/answer.json", {"answer": "fixture"})
    _json(
        path / "artifacts/manifest.json",
        [
            {
                "source": "/app",
                "destination": "artifacts/app",
                "type": "directory",
                "status": "ok",
            }
        ],
    )
    return path


def test_plan_denominator_and_independent_statuses(tmp_path: Path) -> None:
    """预算结束可通过，缺分与有效零分分开，归档异常不改评分。"""
    job = tmp_path / "job"
    _trial(job, "success", "a", reward=1)
    _trial(job, "failure", "a", reward=0)
    _trial(job, "budget", "b", reward=1, stop="budget_exhausted", trace_status="partial")
    _trial(job, "missing", "b", reward=None, exception="RewardFileNotFoundError")

    report = summarize_job(job, task_ids=["a", "b", "c"], attempts=2)
    assert report["counts"] == {
        "planned": 6,
        "started": 4,
        "not_started": 2,
        "valid_scored": 3,
        "passed": 2,
        "exceptions": 1,
        "archive_complete": 3,
        "review_materials_complete": 3,
        "extra_trials": 0,
    }
    assert report["rates"]["planned_success"] == 2 / 6
    assert report["rates"]["scored_success"] == 2 / 3
    assert report["rates"]["score_coverage"] == 3 / 6
    assert report["rates"]["review_materials_coverage"] == 3 / 4
    assert report["pass_at_1"] is None
    assert report["by_task"]["a"]["passed"] == 1
    budget = next(row for row in report["trials"] if row["trial_name"] == "budget")
    assert (budget["execution_status"], budget["reward"], budget["archive_status"]) == (
        "budget_exhausted",
        1,
        "partial",
    )
    assert budget["input_tokens"] == 120
    assert budget["output_tokens"] == 15
    assert budget["cost_usd"] is None
    assert report["metrics"]["cost_usd"] == {"known_sum": None, "known_trials": 0}
    assert report["metrics"]["agent_seconds"]["median"] == 20


@pytest.mark.parametrize("exception", ["VerifierTimeoutError", "EnvironmentBuildError"])
def test_exception_does_not_turn_reported_zero_into_valid_score(
    tmp_path: Path, exception: str
) -> None:
    """保留原始零分与错误证据，待复核评分不进入有效分母。"""
    _trial(tmp_path, "failed", "a", reward=0, exception=exception)
    report = summarize_job(tmp_path, task_ids=["a"], attempts=1)
    row = report["trials"][0]
    assert row["reward"] == 0
    assert row["score_status"] == "review_required"
    assert report["counts"]["valid_scored"] == 0
    assert report["rates"]["scored_success"] is None


def test_provider_error_and_cleanup_failure_remain_visible(tmp_path: Path) -> None:
    """provider 错误、收尾失败与缺失评分各自保留。"""
    path = _trial(
        tmp_path,
        "provider",
        "a",
        reward=0,
        exception="NonZeroAgentExitCodeError",
        execution_error="provider unavailable",
        stop="failed",
    )
    receipt_path = path / "agent/iris/result.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(cleanup_completed=False, cleanup_error="process still running")
    _json(receipt_path, receipt)
    row = summarize_job(tmp_path, task_ids=["a"], attempts=1)["trials"][0]
    assert row["execution_status"] == "execution_error"
    assert row["execution_error"] == "provider unavailable"
    assert row["cleanup_completed"] is False
    assert row["cleanup_error"] == "process still running"
    assert row["score_status"] == "review_required"


def test_incomplete_trial_and_missing_assets(tmp_path: Path) -> None:
    """已创建但无结果的 trial 不当成未启动，附件丢失使归档降为 partial。"""
    _json(tmp_path / "running/config.json", {"task": {"path": "C:\\tasks\\a"}})
    path = _trial(tmp_path, "done", "b", reward=1)
    _json(
        path / "agent/iris/trace.json",
        {"export": {"status": "complete"}, "assets": [{"path": "assets/missing.txt"}]},
    )
    report = summarize_job(tmp_path, task_ids=["a", "b"], attempts=1)
    assert report["counts"]["started"] == 2
    assert report["trials"][0]["execution_status"] == "incomplete"
    assert report["trials"][0]["input_tokens"] is None
    assert report["trials"][1]["archive_status"] == "partial"
    assert "assets/missing.txt" in report["trials"][1]["missing_archive_files"]


def test_incomplete_package_trial_uses_declared_task_name(tmp_path: Path) -> None:
    """未完成的 Harbor 包任务只有 name，仍应归入计划任务而不是报错。"""
    _json(tmp_path / "running/config.json", {"task": {"name": "example/a"}})

    report = summarize_job(tmp_path, task_ids=["example/a"], attempts=1)

    assert report["counts"]["started"] == 1
    assert report["counts"]["not_started"] == 0
    assert report["counts"]["extra_trials"] == 0
    assert report["trials"][0]["task_id"] == "example/a"
    assert report["trials"][0]["execution_status"] == "incomplete"


def test_extra_trial_is_not_selected_for_better_score(tmp_path: Path) -> None:
    """额外重跑单独展示，不覆盖首次计划运行。"""
    _trial(tmp_path, "01-failure", "a", reward=0)
    _trial(tmp_path, "02-rerun", "a", reward=1)
    report = summarize_job(tmp_path, task_ids=["a"], attempts=1)
    assert report["counts"]["passed"] == 0
    assert report["counts"]["extra_trials"] == 1
    assert report["extra_trials"][0]["reward"] == 1


def test_write_report_keeps_empty_cost_and_relative_evidence(tmp_path: Path) -> None:
    """三个离线文件指向原始材料，不改写任务结果。"""
    job = tmp_path / "job"
    path = _trial(job, "done", "a", reward=1)
    before = (path / "result.json").read_bytes()
    report = summarize_job(job, task_ids=["a"], attempts=1)
    output = tmp_path / "report"
    write_report(report, output)
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["trials"][0]["cost_usd"] is None
    assert saved["trials"][0]["trial_path"] == "../job/done"
    assert saved["pass_at_1"] == 1
    assert "1 / 1" in (output / "report.md").read_text(encoding="utf-8")
    assert (output / "trials.csv").is_file()
    assert (path / "result.json").read_bytes() == before


def test_cli_uses_selected_manifest_tasks_and_ignores_job_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """校准子集独立计数，job 汇总不能被误读成 trial 分数。"""
    job = tmp_path / "job"
    _trial(job, "done", "a", reward=0)
    _json(job / "result.json", {"rewards": {"reward": 1}, "n_total_trials": 100})
    manifest = tmp_path / "manifest.json"
    _json(
        manifest,
        {"name": "dev", "dataset_id": "source", "revision": "fixed", "task_ids": ["a", "b"]},
    )
    output = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reporting",
            str(job),
            "--manifest",
            str(manifest),
            "--attempts",
            "1",
            "--task-id",
            "a",
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["counts"]["planned"] == 1
    assert report["counts"]["passed"] == 0
    assert report["dataset"]["revision"] == "fixed"


@pytest.mark.parametrize("missing", ["test-stdout.txt", "reward.txt"])
def test_missing_verifier_material_does_not_claim_review_coverage(
    tmp_path: Path, missing: str
) -> None:
    """Iris 导出完整也不能掩盖评分日志或原始 reward 丢失。"""
    path = _trial(tmp_path, "done", "a", reward=1)
    (path / "verifier" / missing).unlink()
    report = summarize_job(tmp_path, task_ids=["a"], attempts=1)
    row = report["trials"][0]
    assert row["archive_status"] == "complete"
    assert row["review_materials_status"] == "partial"
    assert f"verifier/{missing}" in row["missing_review_files"]
    assert report["rates"]["archive_coverage"] == 1
    assert report["rates"]["review_materials_coverage"] == 0
    assert row["score_status"] == "valid"


def test_artifact_failure_is_unknown_without_missing_source_reason(tmp_path: Path) -> None:
    """Harbor failed 没区分不存在源和下载异常，独立保留 unknown。"""
    path = _trial(tmp_path, "done", "a", reward=1)
    manifest = json.loads((path / "artifacts/manifest.json").read_text())
    manifest.extend(
        {
            "source": source,
            "destination": f"artifacts{source}",
            "type": "file",
            "status": "failed",
        }
        for source in ("/workspace", "/output")
    )
    (path / "artifacts/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = summarize_job(tmp_path, task_ids=["a"], attempts=1)
    row = report["trials"][0]
    assert row["task_artifacts_status"] == "unknown"
    assert row["artifact_entries"] == manifest
    assert row["review_materials_status"] == "complete"
    assert report["rates"]["review_materials_coverage"] == 1


def test_artifact_ok_with_lost_destination_is_partial(tmp_path: Path) -> None:
    """manifest 已确认下载的文件缺失时，可以确定产物归档不完整。"""
    path = _trial(tmp_path, "done", "a", reward=0)
    _json(
        path / "artifacts/manifest.json",
        [
            {
                "source": "/output/answer.json",
                "destination": "artifacts/output/answer.json",
                "type": "file",
                "status": "ok",
            }
        ],
    )
    row = summarize_job(tmp_path, task_ids=["a"], attempts=1)["trials"][0]
    assert row["task_artifacts_status"] == "partial"
    assert row["missing_artifact_files"] == ["artifacts/output/answer.json"]
    assert row["review_materials_status"] == "complete"


def test_missing_artifact_manifest_does_not_claim_complete(tmp_path: Path) -> None:
    """缺 manifest 的任务产物覆盖未知，仍保留轨迹和评分材料覆盖。"""
    path = _trial(tmp_path, "done", "a", reward=0)
    (path / "artifacts/manifest.json").unlink()
    row = summarize_job(tmp_path, task_ids=["a"], attempts=1)["trials"][0]
    assert row["task_artifacts_status"] == "unknown"
    assert row["review_materials_status"] == "complete"
