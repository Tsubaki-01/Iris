"""验证离线汇总保留计划分母、原始评分与独立归档状态。"""

from __future__ import annotations

import csv
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
        "invalid_scored": 0,
        "missing_scores": 3,
        "review_required": 0,
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


def test_partial_rewards_contribute_to_mean_but_not_full_passes(tmp_path: Path) -> None:
    """合法部分分数进入均值，缺分与额外重跑都不参与均值。"""
    job = tmp_path / "job"
    _trial(job, "01-full", "a", reward=1)
    partial = _trial(job, "02-partial", "a", reward=0.5)
    _trial(job, "03-extra", "a", reward=1)
    _trial(job, "zero", "b", reward=0)
    _trial(job, "missing", "b", reward=None)
    before = (partial / "result.json").read_bytes()

    report = summarize_job(job, task_ids=["a", "b", "c"], attempts=2)

    row = next(row for row in report["trials"] if row["trial_name"] == "02-partial")
    assert row["automatic_score_status"] == row["score_status"] == "valid"
    assert row["reward"] == 0.5
    assert report["counts"]["valid_scored"] == 3
    assert report["counts"]["passed"] == 1
    assert report["mean_reward"] == 0.5
    assert report["rates"]["planned_success"] == 1 / 6
    assert report["rates"]["scored_success"] == 1 / 3
    assert report["rates"]["score_coverage"] == 3 / 6
    assert report["by_task"]["a"]["mean_reward"] == 0.75
    assert report["by_task"]["b"]["mean_reward"] == 0
    assert report["by_task"]["c"]["mean_reward"] is None

    output = tmp_path / "report"
    write_report(report, output)
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["mean_reward"] == 0.5
    with (output / "trials.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved_partial = next(row for row in csv.DictReader(stream) if row["reward"] == "0.5")
    assert saved_partial["score_status"] == "valid"
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "有效评分平均 reward：0.5" in markdown
    assert "完全通过" in markdown
    assert "| a | 1 / 2 | 2 | 0.75 |" in markdown
    assert (partial / "result.json").read_bytes() == before


def test_explicit_review_can_accept_partial_reward(tmp_path: Path) -> None:
    """正常 verifier 的部分分数可在执行异常复核后计入，原始分数保持不变。"""
    job = tmp_path / "job"
    _trial(job, "partial", "a", reward=0.4, execution_error="provider output ended")
    reviews_path = tmp_path / "reviews.json"
    _json(reviews_path, {"partial": _review_decision(0.4, "valid")})

    report = summarize_job(job, task_ids=["a"], attempts=1, review_decisions_path=reviews_path)

    row = report["trials"][0]
    assert row["automatic_score_status"] == "review_required"
    assert row["reviewed_score_status"] == row["score_status"] == "valid"
    assert row["reward"] == 0.4
    assert report["mean_reward"] == 0.4
    assert report["counts"]["passed"] == 0
    assert report["pass_at_1"] == 0


@pytest.mark.parametrize("reward", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_reward_cannot_be_counted_as_valid(tmp_path: Path, reward: float) -> None:
    """外部 reward 只有有限的区间内数值可计分，复核不能将其改成有效。"""
    job = tmp_path / "job"
    _trial(job, "invalid", "a", reward=reward)
    report = summarize_job(job, task_ids=["a"], attempts=1)
    assert report["trials"][0]["score_status"] == "review_required"
    assert report["counts"]["valid_scored"] == 0
    assert report["counts"]["review_required"] == 1
    assert report["mean_reward"] is None

    reviews_path = tmp_path / "reviews.json"
    _json(reviews_path, {"invalid": _review_decision(reward, "valid")})
    with pytest.raises(ValueError, match="raw_reward"):
        summarize_job(job, task_ids=["a"], attempts=1, review_decisions_path=reviews_path)


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


def _review_decision(reward: float | None, status: str) -> dict[str, Any]:
    """生成与显式复核文件一致的通用决定。"""
    return {
        "reviewed_score_status": status,
        "raw_reward": reward,
        "category": "reviewed_evidence",
        "reason": "已检查原始执行与评分材料",
        "retry_as_infrastructure": False,
    }


def test_explicit_reviews_change_validity_without_changing_execution_or_reward(
    tmp_path: Path,
) -> None:
    """复核可接受预算失败的零分，也可排除没有运行评分的原始零分。"""
    job = tmp_path / "job"
    budget = _trial(
        job, "budget", "a", reward=0, stop="failed", exception="NonZeroAgentExitCodeError"
    )
    receipt_path = budget / "agent/iris/result.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["result"] = {"error": {"code": "PROVIDER_ERROR", "details": {"status": "length"}}}
    _json(receipt_path, receipt)
    verifier = _trial(job, "verifier", "b", reward=0)
    (verifier / "verifier/test-stdout.txt").write_text("uvx: command not found")
    _trial(job, "success", "c", reward=1)
    reviews_path = tmp_path / "reviews.json"
    rejected = _review_decision(0, "invalid")
    rejected["retry_as_infrastructure"] = True
    _json(reviews_path, {"budget": _review_decision(0, "valid"), "verifier": rejected})
    raw_before = (budget / "result.json").read_bytes()

    report = summarize_job(
        job, task_ids=["a", "b", "c"], attempts=1, review_decisions_path=reviews_path
    )

    rows = {row["trial_name"]: row for row in report["trials"]}
    assert report["counts"]["planned"] == 3
    assert report["counts"]["valid_scored"] == 2
    assert report["counts"]["invalid_scored"] == 1
    assert report["counts"]["missing_scores"] == report["counts"]["review_required"] == 0
    assert report["counts"]["passed"] == 1
    assert report["rates"]["scored_success"] == 1 / 2
    assert report["by_task"]["a"]["valid_scored"] == 1
    assert report["by_task"]["b"]["valid_scored"] == 0
    assert rows["budget"]["automatic_score_status"] == "review_required"
    assert rows["budget"]["reviewed_score_status"] == "valid"
    assert rows["budget"]["score_status"] == "valid"
    assert rows["budget"]["execution_status"] == "execution_error"
    assert rows["budget"]["execution_error"] == receipt["result"]["error"]
    assert rows["verifier"]["automatic_score_status"] == "valid"
    assert rows["verifier"]["reviewed_score_status"] == "invalid"
    assert rows["verifier"]["score_status"] == "invalid"
    assert rows["verifier"]["retry_as_infrastructure"] is True
    assert rows["budget"]["reward"] == rows["verifier"]["reward"] == 0
    assert rows["budget"]["review_category"] == "reviewed_evidence"
    assert rows["budget"]["review_reason"] == "已检查原始执行与评分材料"
    assert rows["success"]["automatic_score_status"] == "valid"
    assert rows["success"]["reviewed_score_status"] is None
    assert rows["success"]["review_reason"] is None
    assert report["unused_review_trials"] == []
    assert (budget / "result.json").read_bytes() == raw_before


@pytest.mark.parametrize(
    ("changes", "reward", "message"),
    [
        ({"reviewed_score_status": "approved"}, 0, "reviewed_score_status"),
        ({"raw_reward": 1}, 0, "raw_reward"),
        ({"raw_reward": None}, None, "valid"),
        ({"reviewed_score_status": "missing"}, 0, "missing"),
        ({"reason": " "}, 0, "reason"),
        ({"retry_as_infrastructure": "false"}, 0, "retry_as_infrastructure"),
    ],
)
def test_invalid_review_decision_is_rejected(
    tmp_path: Path, changes: dict[str, Any], reward: float | None, message: str
) -> None:
    """复核输入在读取边界拒绝无效状态、冲突分数和缺失依据。"""
    job = tmp_path / "job"
    trial = _trial(job, "done", "a", reward=reward)
    decision = _review_decision(0, "valid")
    decision.update(changes)
    reviews_path = tmp_path / "reviews.json"
    _json(reviews_path, {"done": decision})
    before = (trial / "result.json").read_bytes()

    with pytest.raises(ValueError, match=message):
        summarize_job(job, task_ids=["a"], attempts=1, review_decisions_path=reviews_path)

    assert (trial / "result.json").read_bytes() == before


def test_shared_review_file_keeps_unused_entries_and_extra_trials_separate(
    tmp_path: Path,
) -> None:
    """共享复核文件可包含其他 job，额外 trial 的决定不改变计划统计。"""
    job = tmp_path / "job"
    _trial(job, "01-first", "a", reward=0)
    _trial(job, "02-extra", "a", reward=1)
    reviews_path = tmp_path / "reviews.json"
    _json(
        reviews_path,
        {
            "01-first": _review_decision(0, "valid"),
            "02-extra": _review_decision(1, "invalid"),
            "other-job-trial": _review_decision(None, "missing"),
        },
    )

    report = summarize_job(job, task_ids=["a"], attempts=1, review_decisions_path=reviews_path)

    assert report["counts"]["planned"] == report["counts"]["valid_scored"] == 1
    assert report["counts"]["passed"] == 0
    assert report["counts"]["extra_trials"] == 1
    assert report["extra_trials"][0]["reward"] == 1
    assert report["extra_trials"][0]["score_status"] == "invalid"
    assert report["unused_review_trials"] == ["other-job-trial"]


def test_cli_exports_explicit_review_and_keeps_raw_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI 将自动状态、复核依据和采用状态写入三种输出，不改原始结果。"""
    job = tmp_path / "job"
    trial = _trial(job, "done", "a", reward=1)
    before = (trial / "result.json").read_bytes()
    manifest = tmp_path / "manifest.json"
    _json(manifest, {"name": "dev", "dataset_id": "source", "revision": "fixed", "task_ids": ["a"]})
    reviews_path = tmp_path / "reviews.json"
    decision = _review_decision(1, "invalid")
    decision["reason"] = "评分未执行 | 已核对日志\n保留原始分数"
    _json(reviews_path, {"done": decision})
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
            "--review-decisions",
            str(reviews_path),
            "--output",
            str(output),
        ],
    )

    main()

    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["counts"]["valid_scored"] == saved["counts"]["passed"] == 0
    assert saved["trials"][0]["automatic_score_status"] == "valid"
    assert saved["trials"][0]["reviewed_score_status"] == "invalid"
    assert saved["trials"][0]["score_status"] == "invalid"
    assert saved["trials"][0]["review_reason"] == decision["reason"]
    with (output / "trials.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["reward"] == "1"
    assert row["automatic_score_status"] == "valid"
    assert row["reviewed_score_status"] == row["score_status"] == "invalid"
    assert row["review_reason"] == decision["reason"]
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "显式复核" in markdown
    assert "评分未执行" in markdown
    assert "保留原始分数" in markdown
    assert (trial / "result.json").read_bytes() == before
