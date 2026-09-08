"""单次 activation 的 durable 事件收集与同步回调。

Example:
    collector.record(commit.events)
"""

# region imports
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from ..lifecycle import RunEvent

# endregion

logger = logging.getLogger(__name__)


class _RunEventCollector:
    """供 runner 与 commit port 共享的唯一事件去重 owner。"""

    def __init__(self, callback: Callable[[RunEvent], None] | None = None) -> None:
        self.events: list[RunEvent] = []
        self._keys: set[tuple[str, int]] = set()
        self._callback = callback

    def record(self, events: Sequence[RunEvent]) -> None:
        """仅检查本批事件的键，并在首次收集时隔离执行同步回调。"""
        for event in events:
            key = (event.run_id, event.sequence)
            if key in self._keys:
                continue
            self._keys.add(key)
            self.events.append(event)
            if self._callback is not None:
                try:
                    self._callback(event)
                except Exception:
                    logger.exception(
                        "durable event callback 处理失败",
                        extra={"run_id": event.run_id, "sequence": event.sequence},
                    )
