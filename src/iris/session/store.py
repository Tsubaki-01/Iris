"""会话存储接口。"""

from __future__ import annotations

from typing import Protocol


class SessionStore(Protocol):
    """会话历史与运行摘要存储契约。"""

    def save_messages(self, session_id: str, messages: list[dict[str, object]]) -> None:
        """保存会话消息列表。"""

    def load_messages(self, session_id: str) -> list[dict[str, object]]:
        """读取会话消息列表。"""

    def save_run_metadata(self, session_id: str, metadata: dict[str, object]) -> None:
        """保存运行元数据。"""

    def load_run_metadata(self, session_id: str) -> dict[str, object]:
        """读取运行元数据。"""

    def append_tool_event(self, session_id: str, event: dict[str, object]) -> None:
        """追加工具调用或结果摘要。"""

    def load_tool_events(self, session_id: str) -> list[dict[str, object]]:
        """读取工具调用或结果摘要列表。"""


__all__ = ["SessionStore"]
