"""内存会话存储实现。"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import cast

from ..exceptions import IrisSessionError


class InMemorySessionStore:
    """使用进程内字典保存 session 数据。

    该实现适合测试、无持久化运行和调用方显式不需要跨进程恢复的场景。它实现
    `SessionStore` 协议，但不会写入本地文件。
    """

    def __init__(self) -> None:
        self._messages: dict[str, list[dict[str, object]]] = {}
        self._run_metadata: dict[str, dict[str, object]] = {}
        self._tool_events: dict[str, list[dict[str, object]]] = {}
        self._tool_event_payloads: dict[tuple[str, str], str] = {}

    def save_messages(self, session_id: str, messages: list[dict[str, object]]) -> None:
        """保存会话消息列表。"""
        self._messages[session_id] = deepcopy(messages)

    def load_messages(self, session_id: str) -> list[dict[str, object]]:
        """读取会话消息列表。"""
        return deepcopy(self._messages.get(session_id, []))

    def save_run_metadata(self, session_id: str, metadata: dict[str, object]) -> None:
        """保存运行元数据。"""
        self._run_metadata[session_id] = deepcopy(metadata)

    def load_run_metadata(self, session_id: str) -> dict[str, object]:
        """读取运行元数据。"""
        return deepcopy(self._run_metadata.get(session_id, {}))

    def append_tool_event(self, session_id: str, event: dict[str, object]) -> None:
        """追加工具调用或结果摘要。"""
        self._tool_events.setdefault(session_id, []).append(deepcopy(event))

    def append_tool_event_once(
        self,
        session_id: str,
        event_id: str,
        event: dict[str, object],
    ) -> None:
        """按稳定 event ID 幂等追加工具结果事件。"""
        payload, canonical_payload = _normalize_idempotent_event(event_id, event)
        key = (session_id, event_id)
        existing_payload = _find_existing_event_payload(
            self._tool_events.get(session_id, []), event_id
        )
        if existing_payload is None:
            existing_payload = self._tool_event_payloads.get(key)
        if existing_payload is not None:
            if existing_payload != canonical_payload:
                raise IrisSessionError(
                    "相同 event_id 的 tool event payload 不一致",
                    session_id=session_id,
                    event_id=event_id,
                )
            self._tool_event_payloads[key] = existing_payload
            return
        self._tool_events.setdefault(session_id, []).append(deepcopy(payload))
        self._tool_event_payloads[key] = canonical_payload

    def load_tool_events(self, session_id: str) -> list[dict[str, object]]:
        """读取工具调用或结果摘要列表。"""
        return deepcopy(self._tool_events.get(session_id, []))


def _normalize_idempotent_event(
    event_id: str,
    event: dict[str, object],
) -> tuple[dict[str, object], str]:
    if not event_id.strip():
        raise IrisSessionError("tool event_id 不能为空")
    if "event_id" in event and event["event_id"] != event_id:
        raise IrisSessionError("tool event 包含冲突的 event_id", event_id=event_id)
    payload = dict(event)
    payload["event_id"] = event_id
    canonical_payload = _canonical_json(payload)
    return cast(dict[str, object], json.loads(canonical_payload)), canonical_payload


def _find_existing_event_payload(events: list[dict[str, object]], event_id: str) -> str | None:
    payloads = [_canonical_json(event) for event in events if event.get("event_id") == event_id]
    if not payloads:
        return None
    if len(set(payloads)) != 1:
        raise IrisSessionError("相同 event_id 的已有 tool event payload 不一致", event_id=event_id)
    return payloads[0]


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


__all__ = ["InMemorySessionStore"]
