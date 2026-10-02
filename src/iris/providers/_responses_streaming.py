"""将 LiteLLM Responses 事件投影为 Iris 模型流式事件。

原生 SSE 和 LiteLLM 模拟流式使用同一事件契约。增量只用于展示，
最终 response.output 经 mapper 解析后才成为可提交响应。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..exceptions import (
    IrisProviderError,
    IrisProviderStreamInterruptedError,
    IrisProviderStreamProtocolError,
)
from ..message import (
    ModelBlockCompleted,
    ModelBlockDelta,
    ModelBlockRef,
    ModelBlockStarted,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelResponseStarted,
    ModelStreamEvent,
    ModelStreamScope,
    ModelUsageSnapshot,
    ModelUsageUpdated,
)
from ._stream_utils import close_raw_stream, safe_provider_error
from .responses import ResponsesMapper

_AsMapping = Callable[[Any], Mapping[str, Any]]
_ErrorMapper = Callable[[Exception], IrisProviderError]
_TEXT_EVENTS = {
    "response.output_text": ("text", "content_index", "text"),
    "response.refusal": ("text", "content_index", "refusal"),
    "response.reasoning_text": ("thinking", "content_index", "text"),
    "response.reasoning_summary_text": ("thinking", "summary_index", "text"),
}


@dataclass(slots=True)
class _BlockState:
    """一段可展示内容的稳定身份与当前文本。"""

    block: ModelBlockRef
    snapshot: str = ""
    completed: bool = False


class ResponsesStreamAccumulator:
    """消费 Responses 事件，保留完整终态与有序 Iris 展示事件。"""

    def __init__(self, *, scope: ModelStreamScope, as_mapping: _AsMapping) -> None:
        """初始化当前 provider attempt 的展示状态。"""
        self._scope = scope
        self._as_mapping = as_mapping
        self._sequence = 1
        self._response_id = ""
        self._started = False
        self.terminal_emitted = False
        self._semantic_output_emitted = False
        self._blocks: dict[tuple[int, str, int], _BlockState] = {}
        self._tool_items: dict[int, Mapping[str, Any]] = {}

    def feed(self, raw_event: Any) -> tuple[ModelStreamEvent, ...]:
        """完整处理一个 Responses 事件后才向调用方交付其展示事件。"""
        if self.terminal_emitted:
            raise IrisProviderStreamProtocolError("provider terminal 后仍收到 event")
        sequence_before = self._sequence
        semantic_before = self._semantic_output_emitted
        try:
            return self._feed_event(self._as_mapping(raw_event))
        except Exception:
            self._sequence = sequence_before
            self._semantic_output_emitted = semantic_before
            raise

    def _feed_event(self, event: Mapping[str, Any]) -> tuple[ModelStreamEvent, ...]:
        event_type = event.get("type")
        if not isinstance(event_type, str):
            raise IrisProviderStreamProtocolError("Responses event 缺少 type")
        response = self._as_mapping(event.get("response"))
        if response.get("id"):
            self._response_id = response["id"]
        events: list[ModelStreamEvent] = []
        if not self._started:
            self._started = True
            events.append(
                ModelResponseStarted(
                    **self._event_fields(),
                    response_id=self._response_id
                    or (f"{self._scope.model_stream_id}:response:{self._scope.attempt}"),
                )
            )

        if event_type in {"response.completed", "response.failed", "response.incomplete"}:
            if response.get("status") != event_type.removeprefix("response."):
                raise IrisProviderStreamProtocolError("Responses event 与 response.status 不一致")
            parsed = ResponsesMapper().parse_response(response, provider=self._scope.provider)
            events.extend(self._complete_blocks())
            events.append(
                ModelUsageUpdated(
                    **self._event_fields(),
                    usage=ModelUsageSnapshot(
                        input_tokens=parsed.input_tokens,
                        output_tokens=parsed.output_tokens,
                        total_tokens=parsed.total_tokens,
                        complete=True,
                    ),
                )
            )
            self.terminal_emitted = True
            events.append(
                ModelResponseCompleted(
                    **self._event_fields(),
                    response=parsed,
                    semantic_output_emitted=self._semantic_output_emitted,
                )
            )
        elif event_type == "error":
            raise IrisProviderError(
                "Responses stream 返回 error",
                status="error",
                reason=f"{event.get('code') or 'error'}: {event.get('message', '')}",
                **({"usage": event["usage"]} if event.get("usage") is not None else {}),
            )
        elif event_type == "response.output_item.added":
            item = self._as_mapping(event.get("item"))
            if item.get("type") == "function_call":
                output_index = self._output_index(event)
                self._tool_items[output_index] = item
                state = self._tool_state(output_index, events)
                name = item.get("name", "")
                if name:
                    events.append(self._delta(state, "tool_name", name, name))
                # added 只提供工具身份；参数以 delta/done 为准，避免重复累计快照。
        elif event_type in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        }:
            state = self._tool_state(self._output_index(event), events)
            if event_type.endswith(".delta"):
                events.extend(self._append(state, event.get("delta"), "tool_arguments"))
            else:
                events.extend(self._finish_block(state, event.get("arguments"), "tool_arguments"))
        elif event_type == "response.output_item.done":
            events.extend(self._complete_blocks(output_index=self._output_index(event)))
        else:
            prefix, _, suffix = event_type.rpartition(".")
            if prefix in _TEXT_EVENTS and suffix in {"delta", "done"}:
                kind, index_field, text_field = _TEXT_EVENTS[prefix]
                state = self._text_state(event, prefix, kind, index_field, events)
                if suffix == "delta":
                    events.extend(self._append(state, event.get("delta"), kind))
                else:
                    events.extend(self._finish_block(state, event.get(text_field), kind))
        return tuple(events)

    def finish(self) -> tuple[ModelStreamEvent, ...]:
        """EOF 不能替代 Responses 的 response.completed 终态。"""
        if not self.terminal_emitted:
            raise IrisProviderStreamInterruptedError("Responses stream 在原生终态前结束")
        return ()

    def failure_events(self, error: IrisProviderError) -> tuple[ModelStreamEvent, ...]:
        """先交付失败响应已知用量，再交付唯一失败终态。"""
        events: list[ModelStreamEvent] = []
        usage = error.context.get("usage")
        if usage is not None:
            events.append(
                ModelUsageUpdated(
                    **self._event_fields(),
                    usage=ModelUsageSnapshot(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        complete=True,
                    ),
                )
            )
        events.append(self.fail(error))
        return tuple(events)

    def fail(self, error: IrisProviderError) -> ModelResponseFailed:
        """将已归一化的 provider 异常转换为唯一失败终态。"""
        if self.terminal_emitted:
            raise IrisProviderStreamProtocolError("provider terminal 已生成")
        self.terminal_emitted = True
        return ModelResponseFailed(
            **self._event_fields(),
            error=safe_provider_error(error),
            semantic_output_emitted=self._semantic_output_emitted,
        )

    @staticmethod
    def _output_index(event: Mapping[str, Any]) -> int:
        value = event.get("output_index")
        if not isinstance(value, int) or value < 0:
            raise IrisProviderStreamProtocolError("Responses output_index 必须为非负整数")
        return value

    def _text_state(
        self,
        event: Mapping[str, Any],
        prefix: str,
        kind: str,
        index_field: str,
        events: list[ModelStreamEvent],
    ) -> _BlockState:
        key = (self._output_index(event), prefix, event.get(index_field, 0))
        state = self._blocks.get(key)
        if state is None:
            state = self._new_block(
                key,
                kind=kind,
                block_id=f"{event.get('item_id') or key[0]}:{prefix}:{key[2]}",
            )
            events.append(ModelBlockStarted(**self._event_fields(), block=state.block))
        return state

    def _tool_state(self, output_index: int, events: list[ModelStreamEvent]) -> _BlockState:
        key = (output_index, "function_call", 0)
        state = self._blocks.get(key)
        if state is None:
            item = self._tool_items.get(output_index)
            if item is None or not item.get("call_id") or not item.get("id"):
                raise IrisProviderStreamProtocolError("工具参数事件缺少对应 function_call item")
            state = self._new_block(
                key,
                kind="tool_call",
                block_id=item["id"],
                tool_call_id=item["call_id"],
            )
            events.append(ModelBlockStarted(**self._event_fields(), block=state.block))
        return state

    def _new_block(
        self,
        key: tuple[int, str, int],
        *,
        kind: str,
        block_id: str,
        tool_call_id: str | None = None,
    ) -> _BlockState:
        state = _BlockState(
            ModelBlockRef(
                index=len(self._blocks),
                block_id=block_id,
                kind=kind,
                tool_call_id=tool_call_id,
            )
        )
        self._blocks[key] = state
        return state

    def _append(self, state: _BlockState, value: Any, channel: str) -> list[ModelStreamEvent]:
        if not isinstance(value, str):
            raise IrisProviderStreamProtocolError("Responses 文本增量必须是字符串")
        if not value:
            return []
        if state.completed:
            raise IrisProviderStreamProtocolError("已完成内容块后仍收到增量")
        state.snapshot += value
        return [self._delta(state, channel, value, state.snapshot)]

    def _delta(
        self, state: _BlockState, channel: str, delta: str, snapshot: str
    ) -> ModelBlockDelta:
        self._semantic_output_emitted = True
        return ModelBlockDelta(
            **self._event_fields(),
            block=state.block,
            channel=channel,
            delta=delta,
            snapshot=snapshot,
        )

    def _finish_block(
        self,
        state: _BlockState,
        text: Any,
        channel: str,
    ) -> list[ModelStreamEvent]:
        if not isinstance(text, str) or not text.startswith(state.snapshot):
            raise IrisProviderStreamProtocolError("Responses done 内容与已发送增量不一致")
        events = self._append(state, text[len(state.snapshot) :], channel)
        if not state.completed:
            state.completed = True
            events.append(ModelBlockCompleted(**self._event_fields(), block=state.block))
        return events

    def _complete_blocks(self, *, output_index: int | None = None) -> list[ModelStreamEvent]:
        events: list[ModelStreamEvent] = []
        for key, state in self._blocks.items():
            if state.completed or output_index is not None and key[0] != output_index:
                continue
            state.completed = True
            events.append(ModelBlockCompleted(**self._event_fields(), block=state.block))
        return events

    def _event_fields(self) -> dict[str, Any]:
        fields = {
            "scope": self._scope,
            "sequence": self._sequence,
            "occurred_at": datetime.now(UTC),
        }
        self._sequence += 1
        return fields


async def _iter_responses_events(
    raw_stream: AsyncIterator[Any],
    *,
    scope: ModelStreamScope,
    as_mapping: _AsMapping,
    error_mapper: _ErrorMapper,
) -> AsyncGenerator[ModelStreamEvent, None]:
    """拉取 LiteLLM Responses 事件；唯一终态后停止拉取并关闭底层流。"""
    accumulator = ResponsesStreamAccumulator(scope=scope, as_mapping=as_mapping)
    try:
        try:
            async for raw_event in raw_stream:
                for event in accumulator.feed(raw_event):
                    yield event
                if accumulator.terminal_emitted:
                    return
            for event in accumulator.finish():
                yield event
        except IrisProviderError as exc:
            for event in accumulator.failure_events(exc):
                yield event
        except Exception as exc:
            for event in accumulator.failure_events(error_mapper(exc)):
                yield event
    finally:
        await close_raw_stream(raw_stream)
