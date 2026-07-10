"""CLI 侧 provider 调用追踪。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..message import LLMRequest, LLMResponse
from ..runtime import RuntimeProvider


@dataclass(slots=True)
class TraceStep:
    """一次 provider 调用的 request/response 快照。"""

    turn_index: int
    step_index: int
    request: LLMRequest
    response: LLMResponse | None = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float | None = None

    def snapshot(self) -> dict[str, Any]:
        """返回可序列化的 trace 快照。"""
        duration_ms: float | None = None
        if self.finished_at is not None:
            duration_ms = round((self.finished_at - self.started_at) * 1000, 2)
        return {
            "turn_index": self.turn_index,
            "step_index": self.step_index,
            "duration_ms": duration_ms,
            "request": self.request.model_dump(mode="json"),
            "response": (
                self.response.model_dump(mode="json")
                if self.response is not None
                else None
            ),
            "error": self.error,
        }


class ChatTraceStore:
    """保存 chat 进程内 provider trace，并可追加写入 JSONL。"""

    def __init__(self, trace_file: Path | None = None) -> None:
        """初始化 trace store。

        Args:
            trace_file: 可选 JSONL 文件路径。
        """
        self.trace_file = trace_file
        self._turn_index = 0
        self._steps: list[TraceStep] = []
        self.warnings: list[str] = []

    def start_turn(self, turn_index: int) -> None:
        """标记当前用户轮次。"""
        self._turn_index = turn_index

    def append_request(self, request: LLMRequest) -> TraceStep:
        """记录一次 provider request。"""
        step = TraceStep(
            turn_index=self._turn_index,
            step_index=len(self.steps_for_turn(self._turn_index)) + 1,
            request=request,
            started_at=time.time(),
        )
        self._steps.append(step)
        return step

    def attach_response(self, step: TraceStep, response: LLMResponse) -> None:
        """记录 provider response，并在需要时写入 JSONL。"""
        step.response = response
        step.finished_at = time.time()
        self.write_jsonl(step)

    def attach_error(self, step: TraceStep, error: Exception) -> None:
        """记录 provider 异常，并在需要时写入 JSONL。"""
        step.error = f"{error.__class__.__name__}: {error}"
        step.finished_at = time.time()
        self.write_jsonl(step)

    def steps_for_turn(self, turn_index: int) -> list[TraceStep]:
        """返回指定用户轮次的 trace steps。"""
        return [step for step in self._steps if step.turn_index == turn_index]

    def write_jsonl(self, step: TraceStep) -> None:
        """追加写入单条 JSONL trace。"""
        if self.trace_file is None:
            return
        try:
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(step.snapshot(), ensure_ascii=False) + "\n")
        except OSError as exc:
            self.warnings.append(f"trace 文件写入失败: {exc}")


class TracingRuntimeProvider:
    """包装真实 provider，记录每次 `LLMRequest` 和 `LLMResponse`。"""

    def __init__(
        self,
        delegate: RuntimeProvider,
        trace_store: ChatTraceStore,
    ) -> None:
        """创建 provider wrapper。

        Args:
            delegate: 实际 provider。
            trace_store: trace 存储。
        """
        self.delegate = delegate
        self.trace_store = trace_store

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """记录请求并转发给真实 provider。"""
        step = self.trace_store.append_request(request)
        try:
            response = await self.delegate.complete(request)
        except Exception as exc:
            self.trace_store.attach_error(step, exc)
            raise
        self.trace_store.attach_response(step, response)
        return response


__all__ = ["ChatTraceStore", "TraceStep", "TracingRuntimeProvider"]
