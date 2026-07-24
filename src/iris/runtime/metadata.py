"""Runtime run metadata 构造与恢复同步。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ..exceptions import HITLCheckpointInvalidError
from ..message import LLMResponse
from ..session import SessionStore
from .errors import error_result, normalize_runtime_error
from .models import RuntimeErrorInfo, RuntimeStatus, RuntimeTurnResult


def build_run_metadata(
    *,
    existing: dict[str, object],
    session_id: str,
    run_id: str,
    status: RuntimeStatus,
    provider: str,
    response: LLMResponse | None,
    message_count: int,
    metadata: Mapping[str, Any],
    steps: int = 1,
    tool_count: int = 0,
    error: RuntimeErrorInfo | None = None,
    waiting_human: bool = False,
    interaction_id: str | None = None,
) -> dict[str, object]:
    """构建 session 中保存的 append-like run metadata。"""
    latest_run: dict[str, object] = {
        "session_id": session_id,
        "run_id": run_id,
        "status": status.value,
        "provider": cast(object, provider),
        "model": cast(object, response.model if response is not None else ""),
        "finish_reason": cast(
            object,
            response.finish_reason if response is not None else "",
        ),
        "input_tokens": response.input_tokens if response is not None else 0,
        "output_tokens": response.output_tokens if response is not None else 0,
        "total_tokens": response.total_tokens if response is not None else 0,
        "message_count": message_count,
        "steps": steps,
        "tool_count": tool_count,
        "metadata": dict(metadata),
    }
    if error is not None:
        latest_run["error"] = error.model_dump(mode="json")
    if waiting_human:
        latest_run["waiting_human"] = True
    if interaction_id is not None:
        latest_run["interaction_id"] = interaction_id
    runs = existing.get("runs", [])
    run_list = list(runs) if isinstance(runs, list) else []
    run_list.append(latest_run)
    return {**existing, "latest_run": latest_run, "runs": run_list}


def build_resume_run_metadata(
    *,
    existing: dict[str, object],
    result: RuntimeTurnResult,
    message_count: int,
) -> dict[str, object]:
    """基于原 run snapshot 构建 resume 后的 append-like metadata。"""
    previous = existing.get("latest_run")
    latest_run = dict(previous) if isinstance(previous, dict) else {}
    latest_run.update(
        session_id=result.session_id,
        run_id=result.run_id,
        status=result.status.value,
        message_count=message_count,
        steps=result.steps,
        tool_count=len(result.tool_results),
    )
    latest_run.pop("waiting_human", None)
    latest_run.pop("interaction_id", None)
    latest_run.pop("error", None)
    if result.error is not None:
        latest_run["error"] = result.error.model_dump(mode="json")
    if result.status is RuntimeStatus.WAITING_HUMAN:
        if result.pending_interaction is None:
            raise HITLCheckpointInvalidError("waiting_human 结果缺少 pending interaction")
        latest_run["waiting_human"] = True
        latest_run["interaction_id"] = result.pending_interaction.interaction_id
    runs = existing.get("runs", [])
    run_list = list(runs) if isinstance(runs, list) else []
    run_list.append(latest_run)
    return {**existing, "latest_run": latest_run, "runs": run_list}


def synchronize_resume_metadata(
    *,
    session_store: SessionStore,
    result: RuntimeTurnResult,
) -> RuntimeTurnResult:
    """在向 host 返回 resume 结果前同步 latest run snapshot。"""
    try:
        existing = session_store.load_run_metadata(result.session_id)
        metadata = build_resume_run_metadata(
            existing=existing,
            result=result,
            message_count=len(session_store.load_messages(result.session_id)),
        )
        session_store.save_run_metadata(result.session_id, metadata)
    except Exception as exc:
        return error_result(
            session_id=result.session_id,
            run_id=result.run_id,
            error=normalize_runtime_error(exc),
            assistant_message=result.assistant_message,
            steps=result.steps,
            metadata=result.metadata,
        )
    return result


__all__ = ["build_resume_run_metadata", "build_run_metadata", "synchronize_resume_metadata"]
