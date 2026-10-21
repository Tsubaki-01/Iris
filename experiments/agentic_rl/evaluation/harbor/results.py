"""保存任务结果、资源收尾与轨迹状态的独立投影。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iris.harness import AgentRunner
    from iris.lifecycle import RunResult

logger = logging.getLogger(__name__)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """原子写入本次 invocation 的控制或结果文件。"""
    from iris.utils.files import atomic_write_text

    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def save_results(
    *,
    runner: AgentRunner | None,
    session_id: str,
    run_id: str,
    model: str,
    logs_dir: Path,
    result: RunResult | None,
    cleanup_error: BaseException | None,
    execution_error: BaseException | None,
) -> dict[str, Any]:
    """先保存收尾事实，再尽力导出轨迹，不让归档错误改写业务结果。"""
    from iris.harness import collect_trajectory, write_trajectory

    record: dict[str, Any] = {
        "session_id": session_id,
        "run_id": run_id,
        "model": model,
        "admitted": runner is not None and runner.store.load_run(run_id) is not None,
        "phase": result.run.phase.value if result is not None else None,
        "stop_reason": (
            result.run.stop_reason.value
            if result is not None and result.run.stop_reason is not None
            else None
        ),
        "usage": result.run.usage.model_dump(mode="json") if result is not None else None,
        "result": result.model_dump(mode="json") if result is not None else None,
        "execution_error": str(execution_error) if execution_error is not None else None,
        "cleanup_completed": cleanup_error is None,
        "cleanup_error": str(cleanup_error) if cleanup_error is not None else None,
        "trace_status": "failed",
        "trace_errors": [],
    }
    write_json(logs_dir / "result.json", record)
    if runner is not None:
        try:
            trace = collect_trajectory(runner, session_id)
            if cleanup_error is not None:
                trace["export"]["status"] = "partial"
                trace["export"]["errors"].append(
                    {"stage": "cleanup", "message": str(cleanup_error)}
                )
            exported = write_trajectory(trace, logs_dir)
            record["trace_status"] = exported["export"]["status"]
            record["trace_errors"] = exported["export"]["errors"]
            record["configuration"] = exported["configuration"]
        except Exception as error:
            logger.exception("轨迹导出失败，保留原运行结果")
            record["trace_errors"] = [{"stage": "collect", "message": str(error)}]
    else:
        record["trace_errors"] = [{"stage": "setup", "message": "Runner 尚未创建"}]
    write_json(logs_dir / "result.json", record)
    return record


def token_usage(record: dict[str, Any]) -> tuple[int | None, int | None]:
    """主模型与压缩用量各计一次，没有 SDK 用量时保留未知。"""
    usage = record["usage"]
    if usage is None:
        return None, None
    return (
        usage["input_tokens"] + usage["compaction"]["input_tokens"],
        usage["output_tokens"] + usage["compaction"]["output_tokens"],
    )
