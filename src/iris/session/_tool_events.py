"""Session tool event 幂等追加规则。"""

from __future__ import annotations

import json
from typing import cast

from ..exceptions import IrisSessionError


def prepare_tool_event_append(
    existing_events: list[dict[str, object]],
    event: dict[str, object],
) -> dict[str, object] | None:
    """校验 event，并判断是否需要追加规范化 payload。"""
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        raise IrisSessionError("tool event_id 必须是非空字符串")

    canonical_payload = _canonical_json(event)
    payload = cast(dict[str, object], json.loads(canonical_payload))
    existing_payloads = [
        _canonical_json(existing)
        for existing in existing_events
        if existing.get("event_id") == event_id
    ]
    if not existing_payloads:
        return payload
    if len(set(existing_payloads)) != 1:
        raise IrisSessionError(
            "相同 event_id 的已有 tool event payload 不一致",
            event_id=event_id,
        )
    if existing_payloads[0] != canonical_payload:
        raise IrisSessionError(
            "相同 event_id 的 tool event payload 不一致",
            event_id=event_id,
        )
    return None


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise IrisSessionError("Session 数据必须可 JSON 序列化") from exc


__all__ = ["prepare_tool_event_append"]
