"""Trusted live facts 到 remote-safe payload 的 allowlist projection。

Projection 不分配 live sequence，也不保存状态；broker 是唯一 order owner。

Example:
    projected = project_live_fact(fact)
"""

# region imports

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, assert_never, cast

from pydantic import JsonValue, TypeAdapter

from ..goal.models import GoalChanged, GoalView
from ..harness.configuration import ConfigurationApplied
from ..harness.control import SessionControlSnapshot
from ..harness.maintenance_models import MaintenanceChanged, ResourceMaintenanceView
from ..harness.streaming import (
    CommandCleanupFailed,
    LineagedLiveFact,
    LiveFact,
    SessionControlChanged,
    SessionSubmissionEvent,
    SubagentLinked,
    UnscopedLiveFact,
)
from ..lifecycle import RunEvent
from ..lifecycle.history import RunLineage
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
    TextBlock,
)
from ..observability.facts import SourceAdopted
from ..runtime import RuntimeStreamEvent
from ..runtime.diagnostics import ContextPreparation
from ..tools import ToolResult

# endregion

_GOAL_VIEW_ADAPTER = TypeAdapter(GoalView)
_CONTROL_ADAPTER = TypeAdapter(SessionControlSnapshot)
_LINEAGE_ADAPTER = TypeAdapter(RunLineage)
_MAINTENANCE_ADAPTER = TypeAdapter(ResourceMaintenanceView)


@dataclass(frozen=True, slots=True)
class _ProjectedLiveFact:
    """Broker 分配 sequence 前的一条 remote-safe scope fact。"""

    scope: Literal["run", "session", "session_tree", "resource"]
    scope_id: str
    kind: str
    run_id: str | None
    session_id: str | None
    activation_id: str | None
    durable_sequence: int | None
    payload: dict[str, JsonValue]
    critical: bool
    coalescing_key: tuple[str, ...] | None = None
    lineage: RunLineage | None = None


def project_live_fact(fact: LiveFact) -> tuple[_ProjectedLiveFact, ...]:
    """原 scope 保持精确身份，额外投影到根会话树。"""
    lineage = fact.lineage if isinstance(fact, LineagedLiveFact) else None
    original = fact.fact if isinstance(fact, LineagedLiveFact) else fact
    projected = _project_fact(original)
    output: list[_ProjectedLiveFact] = []
    for item in projected:
        item = replace(item, lineage=lineage)
        output.append(item)
        if item.scope == "session":
            output.append(
                replace(
                    item,
                    scope="session_tree",
                    scope_id=lineage.root_session_id if lineage is not None else item.scope_id,
                )
            )
    return tuple(output)


def _project_fact(fact: UnscopedLiveFact) -> tuple[_ProjectedLiveFact, ...]:
    """把一条 trusted fact 投影到一个或两个 remote scopes。

    Args:
        fact (LiveFact): Runtime、durable 或 session submission fact。

    Returns:
        tuple[_ProjectedLiveFact, ...]: Allowlisted run/session projections。
    """
    if isinstance(fact, MaintenanceChanged):
        return (
            _ProjectedLiveFact(
                scope="resource",
                scope_id=fact.resource.resource_ref,
                kind="maintenance.changed",
                run_id=None,
                session_id=None,
                activation_id=None,
                durable_sequence=None,
                payload={
                    "coordinator_id": fact.coordinator_id,
                    "revision": fact.revision,
                    "foreground_count": fact.foreground_count,
                    "resource": _MAINTENANCE_ADAPTER.dump_python(fact.resource, mode="json"),
                },
                critical=False,
                coalescing_key=("maintenance", fact.resource.resource_ref),
            ),
        )
    if isinstance(fact, ConfigurationApplied):
        return _run_and_session(
            _ProjectedLiveFact(
                scope="run",
                scope_id=fact.run_id,
                kind="configuration.applied",
                run_id=fact.run_id,
                session_id=fact.session_id,
                activation_id=fact.activation_id,
                durable_sequence=None,
                payload={
                    "configuration_snapshot_id": fact.configuration_snapshot_id,
                    "agent_id": fact.agent_id,
                },
                critical=True,
            )
        )
    if isinstance(fact, SourceAdopted):
        projected = _ProjectedLiveFact(
            scope="resource" if fact.resource_ref is not None else "run",
            scope_id=fact.resource_ref if fact.resource_ref is not None else cast(str, fact.run_id),
            kind="source.adopted",
            run_id=fact.run_id,
            session_id=fact.session_id,
            activation_id=fact.activation_id,
            durable_sequence=None,
            payload={
                "adoption_id": fact.adoption_id,
                "source_kind": fact.source_kind,
                "owner_kind": fact.owner_kind,
                "adoption_boundary": fact.adoption_boundary,
                "step_index": fact.step_index,
                "preparation_id": fact.preparation_id,
                "maintenance_cycle_id": fact.maintenance_cycle_id,
                "document_ids": [document.document_id for document in fact.documents],
            },
            critical=False,
        )
        return (projected,) if fact.resource_ref is not None else _run_and_session(projected)
    if isinstance(fact, ContextPreparation):
        return _run_and_session(
            _ProjectedLiveFact(
                scope="run",
                scope_id=fact.run_id,
                kind="context.preparation",
                run_id=fact.run_id,
                session_id=fact.session_id,
                activation_id=fact.activation_id,
                durable_sequence=None,
                payload={
                    "preparation_id": fact.preparation_id,
                    "phase": fact.phase,
                    "step_index": fact.step_index,
                    "configuration_snapshot_id": fact.configuration_snapshot_id,
                    "stage_count": len(fact.stages),
                    "final_input_tokens": fact.final_input_tokens,
                },
                critical=False,
                coalescing_key=(
                    fact.run_id,
                    fact.activation_id,
                    "context.preparation",
                    str(fact.step_index),
                ),
            )
        )
    if isinstance(fact, SubagentLinked):
        lineage = fact.lineage
        return _run_and_session(
            _ProjectedLiveFact(
                scope="run",
                scope_id=lineage.child_run_id,
                kind="subagent.linked",
                run_id=lineage.child_run_id,
                session_id=lineage.child_session_id,
                activation_id=None,
                durable_sequence=None,
                payload={"lineage": _LINEAGE_ADAPTER.dump_python(lineage, mode="json")},
                critical=True,
            )
        )
    if isinstance(fact, RunEvent):
        projected = _project_run_event(fact)
        return _run_and_session(projected)
    if isinstance(fact, SessionSubmissionEvent):
        return (_project_submission_event(fact),)
    if isinstance(fact, SessionControlChanged):
        snapshot = fact.snapshot
        return (
            _ProjectedLiveFact(
                scope="session",
                scope_id=snapshot.session_id,
                kind="session.control.changed",
                run_id=snapshot.current_run_id,
                session_id=snapshot.session_id,
                activation_id=None,
                durable_sequence=None,
                payload={"snapshot": _CONTROL_ADAPTER.dump_python(snapshot, mode="json")},
                critical=True,
            ),
        )
    if isinstance(fact, GoalChanged):
        return (
            _ProjectedLiveFact(
                scope="session",
                scope_id=fact.session_id,
                kind="goal.changed",
                run_id=None,
                session_id=fact.session_id,
                activation_id=None,
                durable_sequence=None,
                payload={"view": _GOAL_VIEW_ADAPTER.dump_python(fact.view, mode="json")},
                critical=False,
                coalescing_key=("goal", fact.session_id),
            ),
        )
    if isinstance(fact, RuntimeStreamEvent):
        projected = _project_runtime_event(fact)
        return _run_and_session(projected)
    if isinstance(fact, CommandCleanupFailed):
        return _run_and_session(
            _ProjectedLiveFact(
                scope="run",
                scope_id=fact.run_id,
                kind="command.cleanup.failed",
                run_id=fact.run_id,
                session_id=fact.session_id,
                activation_id=None,
                durable_sequence=None,
                payload={
                    "error": {
                        "code": fact.error.code,
                        "source": fact.error.source,
                        "message": fact.error.message,
                    }
                },
                critical=True,
            )
        )
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
    if event.kind in {
        "model.step.started",
        "context.compaction.started",
        "context.compaction.completed",
        "context.compaction.failed",
    }:
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
        "content": [
            block.text
            if isinstance(block, TextBlock)
            else f"[image: {block.name}]"
            if block.name
            else "[image]"
            for block in result.content
        ],
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
    if result.tool_name == "edit_file" and not result.is_error and "file_change" in result.data:
        file_change = result.data["file_change"]
        payload["file_change"] = {
            "file_path": file_change["file_path"],
            "patch": file_change["patch"],
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
