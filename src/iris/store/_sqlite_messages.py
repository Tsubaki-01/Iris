"""SQLite 完整会话历史与截点前缀共用的消息行解码。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from ..exceptions import IrisRunPersistenceError
from ..message.message import Msg


def decode_session_messages(
    rows: Sequence[sqlite3.Row],
    *,
    expected_count: int,
    path: Path,
    operation: str,
) -> list[Msg]:
    """在持久化读取边界校验消息数量、连续序号并解码消息。

    Args:
        rows: 按 ordinal 升序读取的完整历史或指定前缀。
        expected_count: 对应 session metadata 或终态截点中的消息数。
        path: 用于错误上下文的数据库路径。
        operation: 用于错误上下文的 store 操作名。

    Returns:
        从 durable rows 独立解码的消息列表。

    Raises:
        IrisRunPersistenceError: 消息数量、序号或持久化内容不满足当前契约。
    """
    try:
        if len(rows) != expected_count:
            raise ValueError("session message_count 与 row count 不一致")
        messages: list[Msg] = []
        for expected_ordinal, row in enumerate(rows, start=1):
            if row["ordinal"] != expected_ordinal:
                raise ValueError("session message ordinal 不连续")
            payload = json.loads(row["message_json"])
            if not isinstance(payload, dict):
                raise TypeError("session message JSON 必须是 object")
            messages.append(Msg.from_dict(payload))
        return messages
    except (
        ValidationError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        json.JSONDecodeError,
    ) as exc:
        raise IrisRunPersistenceError(
            "lifecycle SQLite session history 无法验证",
            path=str(path),
            operation=operation,
        ) from exc
