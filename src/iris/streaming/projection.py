"""Trusted live facts 到 remote-safe payload 的 allowlist projection。

Projection 不分配 live sequence，也不保存状态；broker 是唯一 order owner。

Example:
    projected = project_live_fact(fact)
"""

# region imports

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never, cast

from pydantic import JsonValue

from ..harness.streaming import LiveFact, SessionSubmissionEvent
from ..lifecycle import RunEvent
from ..message import (
    ModelBlockCompleted,
    ModelBlockDelta,
    ModelBlockRef,
    ModelBlockStarted,
    ModelResponseCancelled,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelResponseStarted,
    ModelStreamEvent,
    ModelUsageUpdated,
)
from ..runtime import RuntimeStreamEvent
from ..tools import ToolResult

# endregion


@dataclass(frozen=True, slots=True)
class _ProjectedLiveFact:
    """Broker 分配 sequence 前的一条 remote-safe scope fact。"""

    scope: Literal["run", "session"]
    scope_id: str
    kind: str
    run_id: str | None
    session_id: str | None
    activation_id: str | None
    durable_sequence: int | None
    payload: dict[str, JsonValue]
    critical: bool
    coalescing_key: tuple[str, ...] | None = None


def project_live_fact(fact: LiveFact) -> tuple[_ProjectedLiveFact, ...]:
    """把一条 trusted fact 投影到一个或两个 remote scopes。

    Args:
        fact (LiveFact): Runtime、durable 或 session submission fact。

    Returns:
        tuple[_ProjectedLiveFact, ...]: Allowlisted run/session projections。
    """
    if isinstance(fact, RunEvent):
        projected = _project_run_event(fact)
        return _run_and_session(projected)
    if isinstance(fact, SessionSubmissionEvent):
        return (_project_submission_event(fact),)
    if isinstance(fact, RuntimeStreamEvent):
        projected = _project_runtime_event(fact)
        return _run_and_session(projected)
    assert_never(fact)


def _run_and_session(
    run_fact: _ProjectedLiveFact,
) -> tuple[_ProjectedLiveFact, _ProjectedLiveFact]:
    """复制同一 allowlisted payload 到独立 run/session scopes。"""
    return (
        run_fact,
        _ProjectedLiveFact(
            scope="session",
            scope_id=cast(str, run_fact.session_id),
            kind=run_fact.kind,
            run_id=run_fact.run_id,
            session_id=run_fact.session_id,
            activation_id=run_fact.activation_id,
            durable_sequence=run_fact.durable_sequence,
            payload=run_fact.payload,
            critical=run_fact.critical,
            coalescing_key=run_fact.coalescing_key,
        ),
    )


def _project_run_event(event: RunEvent) -> _ProjectedLiveFact:
    """投影 durable event 的稳定字段与已知安全 payload keys。"""
    payload: dict[str, JsonValue] = {"occurred_at": event.occurred_at.isoformat()}
    if event.step_index is not None:
        payload["step_index"] = event.step_index
    if event.correlation_id is not None:
        payload["correlation_id"] = event.correlation_id
    for key in ("reason", "stop_reason"):
        if key in event.payload:
            payload[key] = cast(JsonValue, event.payload[key])
    return _ProjectedLiveFact(
        scope="run",
        scope_id=event.run_id,
        kind=event.kind.value,
        run_id=event.run_id,
        session_id=event.session_id,
        activation_id=event.activation_id,
        durable_sequence=event.sequence,
        payload=payload,
        critical=True,
    )


def _project_submission_event(event: SessionSubmissionEvent) -> _ProjectedLiveFact:
    """投影 process-local submission identity/status。"""
    submission = event.event
    payload: dict[str, JsonValue] = {
        "submission_id": submission.submission_id,
        "mode": submission.mode,
        "state": submission.state,
    }
    if submission.reason is not None:
        payload["reason"] = submission.reason
    pending = submission.state == "pending"
    return _ProjectedLiveFact(
        scope="session",
        scope_id=event.session_id,
        kind=f"submission.{submission.state}",
        run_id=submission.run_id,
        session_id=event.session_id,
        activation_id=None,
        durable_sequence=None,
        payload=payload,
        critical=not pending,
        coalescing_key=("submission", submission.submission_id) if pending else None,
    )


def _project_runtime_event(event: RuntimeStreamEvent) -> _ProjectedLiveFact:
    """投影 runtime model/tool live event。"""
    if event.kind == "model.step.started":
        return _runtime_projection(
            event,
            kind=event.kind,
            payload={"step_index": event.step_index},
            critical=True,
        )
    if event.kind == "model.event":
        return _project_model_event(event, cast(ModelStreamEvent, event.model_event))
    if event.kind == "tool.preparing":
        return _runtime_projection(
            event,
            kind=event.kind,
            payload=_tool_identity_payload(event),
            critical=False,
            coalescing_key=(
                event.run_id,
                event.activation_id,
                str(event.step_index),
                event.kind,
                cast(str, event.tool_call_id),
            ),
        )
    if event.kind == "tool.started":
        return _runtime_projection(
            event,
            kind=event.kind,
            payload=_tool_identity_payload(event),
            critical=True,
        )
    if event.kind == "tool.completed":
        payload = _tool_identity_payload(event)
        payload.update(_safe_tool_result(cast(ToolResult, event.tool_result)))
        return _runtime_projection(
            event,
            kind=event.kind,
            payload=payload,
            critical=True,
        )
    assert_never(event.kind)


def _project_model_event(
    runtime_event: RuntimeStreamEvent,
    event: ModelStreamEvent,
) -> _ProjectedLiveFact:
    """投影 provider-neutral model event，不透传 raw response。"""
    base: dict[str, JsonValue] = {
        "model_stream_id": event.scope.model_stream_id,
        "provider_sequence": event.sequence,
        "occurred_at": event.occurred_at.isoformat(),
    }
    if isinstance(event, ModelResponseStarted):
        base["response_id"] = event.response_id
        return _runtime_projection(
            runtime_event,
            kind="model.response.started",
            payload=base,
            critical=True,
        )
    if isinstance(event, ModelBlockStarted):
        base.update(_block_payload(event.block))
        return _runtime_projection(
            runtime_event,
            kind="model.block.started",
            payload=base,
            critical=True,
        )
    if isinstance(event, ModelBlockDelta):
        base.update(_block_payload(event.block))
        base.update(
            {
                "channel": event.channel,
                "delta": event.delta,
                "snapshot": event.snapshot,
            }
        )
        return _runtime_projection(
            runtime_event,
            kind="model.block.delta",
            payload=base,
            critical=False,
            coalescing_key=(
                runtime_event.run_id,
                runtime_event.activation_id,
                str(runtime_event.step_index),
                "model.block.delta",
                event.block.block_id,
                event.channel,
            ),
        )
    if isinstance(event, ModelBlockCompleted):
        base.update(_block_payload(event.block))
        return _runtime_projection(
            runtime_event,
            kind="model.block.completed",
            payload=base,
            critical=True,
        )
    if isinstance(event, ModelUsageUpdated):
        base["usage"] = _usage_payload(
            event.usage.input_tokens,
            event.usage.output_tokens,
            event.usage.total_tokens,
            complete=event.usage.complete,
        )
        return _runtime_projection(
            runtime_event,
            kind="model.usage.updated",
            payload=base,
            critical=False,
            coalescing_key=(
                runtime_event.run_id,
                runtime_event.activation_id,
                str(runtime_event.step_index),
                "model.usage.updated",
            ),
        )
    if isinstance(event, ModelResponseCompleted):
        response = event.response
        base.update(
            {
                "provider": response.provider,
                "model": response.model,
                "response_id": response.id,
                "finish_reason": response.finish_reason,
                "usage": _usage_payload(
                    response.input_tokens,
                    response.output_tokens,
                    response.total_tokens,
                ),
                "semantic_output_emitted": event.semantic_output_emitted,
            }
        )
        return _runtime_projection(
            runtime_event,
            kind="model.response.completed",
            payload=base,
            critical=True,
        )
    if isinstance(event, ModelResponseFailed):
        base.update(
            {
                "error": {
                    "code": event.error.code,
                    "message": event.error.message,
                    "retryable": event.error.retryable,
                },
                "semantic_output_emitted": event.semantic_output_emitted,
            }
        )
        return _runtime_projection(
            runtime_event,
            kind="model.response.failed",
            payload=base,
            critical=True,
        )
    if isinstance(event, ModelResponseCancelled):
        if event.error is not None:
            base["error"] = {
                "code": event.error.code,
                "message": event.error.message,
                "retryable": event.error.retryable,
            }
        base["semantic_output_emitted"] = event.semantic_output_emitted
        return _runtime_projection(
            runtime_event,
            kind="model.response.cancelled",
            payload=base,
            critical=True,
        )
    assert_never(event)


def _runtime_projection(
    event: RuntimeStreamEvent,
    *,
    kind: str,
    payload: dict[str, JsonValue],
    critical: bool,
    coalescing_key: tuple[str, ...] | None = None,
) -> _ProjectedLiveFact:
    """构造单个 run-scope runtime projection。"""
    return _ProjectedLiveFact(
        scope="run",
        scope_id=event.run_id,
        kind=kind,
        run_id=event.run_id,
        session_id=event.session_id,
        activation_id=event.activation_id,
        durable_sequence=None,
        payload=payload,
        critical=critical,
        coalescing_key=coalescing_key,
    )


def _block_payload(block: ModelBlockRef) -> dict[str, JsonValue]:
    """投影稳定 block identity。"""
    payload: dict[str, JsonValue] = {
        "block_index": block.index,
        "block_id": block.block_id,
        "block_kind": block.kind,
    }
    if block.tool_call_id is not None:
        payload["tool_call_id"] = block.tool_call_id
    return payload


def _tool_identity_payload(event: RuntimeStreamEvent) -> dict[str, JsonValue]:
    """投影 runtime 已构造的完整 tool identity。"""
    return {
        "tool_call_id": cast(str, event.tool_call_id),
        "tool_name": cast(str, event.tool_name),
        "tool_ordinal": cast(int, event.tool_ordinal),
    }


def _safe_tool_result(result: ToolResult) -> dict[str, JsonValue]:
    """只投影可远端暴露的 ToolResult allowlist。"""
    payload: dict[str, JsonValue] = {
        "content": [block.text for block in result.content],
        "is_error": result.is_error,
    }
    if result.error is not None:
        payload["error"] = {
            "code": result.error.code,
            "message": result.error.message,
            "retryable": result.error.retryable,
        }
    if result.artifact is not None:
        payload["artifact"] = {
            "mime_type": result.artifact.mime_type,
            "size_bytes": result.artifact.size_bytes,
            "preview": result.artifact.preview,
        }
    return payload


def _usage_payload(
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    *,
    complete: bool | None = None,
) -> dict[str, JsonValue]:
    """投影稳定 token usage fields。"""
    usage: dict[str, JsonValue] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    if complete is not None:
        usage["complete"] = complete
    return usage


__all__ = ["project_live_fact"]
