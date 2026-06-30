"""内存会话存储实现。"""

from __future__ import annotations

from copy import deepcopy


class InMemorySessionStore:
    """使用进程内字典保存 session 数据。

    该实现适合测试、无持久化运行和调用方显式不需要跨进程恢复的场景。它实现与
    `SQLiteSessionStore` 相同的 `SessionStore` 协议，但不会写入本地文件。
    """

    def __init__(self) -> None:
        self._messages: dict[str, list[dict[str, object]]] = {}
        self._run_metadata: dict[str, dict[str, object]] = {}
        self._tool_events: dict[str, list[dict[str, object]]] = {}

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

    def load_tool_events(self, session_id: str) -> list[dict[str, object]]:
        """读取工具调用或结果摘要列表。"""
        return deepcopy(self._tool_events.get(session_id, []))


__all__ = ["InMemorySessionStore"]
