"""LiteLLM Chat Completion 流式响应归一化。

该内部模块直接拉取 raw chunks，并在 provider 边界内完成block聚合和终态构造。

Example:
    `ProviderClient.stream()` 使用本模块并只向调用方产出 `ModelStreamEvent`。
"""

# region imports

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from ..exceptions import (
    IrisAPIConnectionError,
    IrisProviderError,
    IrisProviderStreamError,
    IrisProviderStreamInterruptedError,
    IrisProviderStreamProtocolError,
    IrisRateLimitExceededError,
)
from ..message import (
    ContentBlock,
    LLMResponse,
    ModelBlockCompleted,
    ModelBlockDelta,
    ModelBlockRef,
    ModelBlockStarted,
    ModelResponseCancelled,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelResponseStarted,
    ModelStreamEvent,
    ModelStreamScope,
    ModelUsageSnapshot,
    ModelUsageUpdated,
    ProviderStreamError,
    TextBlock,
    ToolUseBlock,
)

# endregion

# ==========================================
#                 内部常量
# ==========================================
# region constants

_logger = logging.getLogger(__name__)

_AsMapping = Callable[[Any], Mapping[str, Any]]
_ErrorMapper = Callable[[Exception], IrisProviderError]

# endregion


@dataclass(slots=True)
class _BlockState:
    """维护单个响应块在当前 provider attempt 内的累积状态。"""

    block: ModelBlockRef
    snapshot: str = ""
    tool_name: str = ""
    tool_arguments: str = ""
    parsed_arguments: dict[str, Any] = field(default_factory=dict)
    completed: bool = False


class ModelStreamAccumulator:
    """将 LiteLLM-shaped chunks 聚合为稳定模型流式事件。"""

    # ==========================================
    #                 状态初始化
    # ==========================================
    # region

    def __init__(
        self,
        *,
        scope: ModelStreamScope,
        as_mapping: _AsMapping,
    ) -> None:
        """初始化一次 provider stream 的内存状态。

        Args:
            scope: 本次 provider attempt 的稳定标识。
            as_mapping: 将 LiteLLM 对象转换为只读 mapping 的边界函数。
        """
        self._scope = scope
        self._as_mapping = as_mapping
        self._next_sequence = 1
        self._next_block_index = 0
        self._response_id = ""
        self._model = ""
        self._raw_object = ""
        self._finish_reason = ""
        self._usage = ModelUsageSnapshot()
        self._started = False
        self._terminal_emitted = False
        self._semantic_output_emitted = False
        self._blocks: dict[tuple[str, int], _BlockState] = {}

    # endregion

    # ==========================================
    #                 公开接口
    # ==========================================
    # region

    def feed(self, raw_chunk: Any) -> tuple[ModelStreamEvent, ...]:
        """消费一个 raw chunk 并返回对应的连续 typed events。

        Args:
            raw_chunk: LiteLLM 返回的单个 Chat Completion chunk。

        Returns:
            本 chunk 产生的零个或多个 provider-neutral events。

        Raises:
            IrisProviderStreamProtocolError: chunk shape、顺序或block identity无效。
        """
        if self._terminal_emitted:
            raise IrisProviderStreamProtocolError("provider terminal 后仍收到 chunk")

        sequence_before = self._next_sequence
        semantic_output_before = self._semantic_output_emitted
        try:
            return self._feed_chunk(raw_chunk)
        except Exception:
            # 一个raw chunk只在完整通过协议校验后才对consumer可见。
            self._next_sequence = sequence_before
            self._semantic_output_emitted = semantic_output_before
            raise

    def _feed_chunk(self, raw_chunk: Any) -> tuple[ModelStreamEvent, ...]:
        """处理单个chunk；协议失败时由`feed()`回滚可观察计数。"""

        chunk = self._as_mapping(raw_chunk)
        if not chunk:
            raise IrisProviderStreamProtocolError("provider chunk 不是可解析的 mapping")

        self._merge_response_identity(chunk)
        choices = self._choices(chunk)
        usage = self._usage_mapping(chunk)
        events: list[ModelStreamEvent] = []

        if not self._started:
            self._started = True
            events.append(
                ModelResponseStarted(
                    **self._event_fields(),
                    response_id=self._response_id_or_fallback(),
                )
            )

        if choices:
            choice = choices[0]
            delta = self._delta(choice)
            finish_reason = choice.get("finish_reason")
            delta_events = self._consume_delta(delta)
            # LiteLLM usage 尾包可能保留字段全为空的占位 choice。
            if self._finish_reason:
                if delta_events or (finish_reason is not None and finish_reason != ""):
                    raise IrisProviderStreamProtocolError("finish reason 后不得再出现语义 choice")
            else:
                events.extend(delta_events)
                if finish_reason is not None and finish_reason != "":
                    if not isinstance(finish_reason, str):
                        raise IrisProviderStreamProtocolError("finish_reason 必须是字符串")
                    self._finish_reason = finish_reason
                    events.extend(self._complete_blocks())

        if usage is not None:
            self._usage = self._usage_from_mapping(usage, complete=False)
            events.append(ModelUsageUpdated(**self._event_fields(), usage=self._usage))
        return tuple(events)

    def finish(self) -> tuple[ModelStreamEvent, ...]:
        """在 raw iterator EOF 后生成合法 terminal。

        Returns:
            最终完整usage事件和唯一response terminal。

        Raises:
            IrisProviderStreamInterruptedError: EOF前没有合法finish reason。
        """
        if self._terminal_emitted:
            raise IrisProviderStreamProtocolError("provider terminal 已生成")
        if not self._finish_reason:
            raise IrisProviderStreamInterruptedError("provider stream 在 finish reason 前结束")

        completed_usage = self._usage.model_copy(update={"complete": True})
        self._usage = completed_usage
        events: list[ModelStreamEvent] = [
            ModelUsageUpdated(**self._event_fields(), usage=completed_usage)
        ]
        self._terminal_emitted = True
        if self._finish_reason in {"cancelled", "canceled"}:
            events.append(
                ModelResponseCancelled(
                    **self._event_fields(),
                    semantic_output_emitted=self._semantic_output_emitted,
                )
            )
            return tuple(events)

        events.append(
            ModelResponseCompleted(
                **self._event_fields(),
                response=self._build_response(),
                semantic_output_emitted=self._semantic_output_emitted,
            )
        )
        return tuple(events)

    def fail(self, error: IrisProviderError) -> ModelResponseFailed:
        """把 provider异常转换为唯一安全失败terminal。

        Args:
            error: 已归一化到 Iris provider领域的异常。

        Returns:
            不包含raw异常详情的失败事件。
        """
        if self._terminal_emitted:
            raise IrisProviderStreamProtocolError("provider terminal 已生成")
        self._terminal_emitted = True
        return ModelResponseFailed(
            **self._event_fields(),
            error=_safe_provider_error(error),
            semantic_output_emitted=self._semantic_output_emitted,
        )

    # endregion

    # ==========================================
    #                 内部辅助
    # ==========================================
    # region

    def _merge_response_identity(self, chunk: Mapping[str, Any]) -> None:
        """合并并校验跨chunk稳定的response identity。"""
        for field_name, attribute_name in (
            ("id", "_response_id"),
            ("model", "_model"),
            ("object", "_raw_object"),
        ):
            value = chunk.get(field_name)
            if value is None or value == "":
                continue
            if not isinstance(value, str):
                raise IrisProviderStreamProtocolError(f"{field_name} 必须是字符串")
            current = getattr(self, attribute_name)
            if current and current != value:
                raise IrisProviderStreamProtocolError(f"{field_name} 在chunks之间不一致")
            setattr(self, attribute_name, value)

    def _choices(self, chunk: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        """读取且校验单choice Chat Completion stream。"""
        raw_choices = chunk.get("choices", [])
        if raw_choices is None:
            return []
        if not isinstance(raw_choices, list):
            raise IrisProviderStreamProtocolError("choices 必须是list")
        if len(raw_choices) > 1:
            raise IrisProviderStreamProtocolError("Iris只支持单choice stream")
        choices = [self._as_mapping(item) for item in raw_choices]
        if any(not item for item in choices):
            raise IrisProviderStreamProtocolError("choice 不是可解析的mapping")
        if choices:
            choice_index = choices[0].get("index", 0)
            if choice_index != 0:
                raise IrisProviderStreamProtocolError("stream choice index 必须为0")
        return choices

    def _usage_mapping(self, chunk: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """读取可选usage mapping。"""
        raw_usage = chunk.get("usage")
        if raw_usage is None:
            return None
        usage = self._as_mapping(raw_usage)
        if not usage:
            raise IrisProviderStreamProtocolError("usage 不是可解析的mapping")
        return usage

    def _delta(self, choice: Mapping[str, Any]) -> Mapping[str, Any]:
        """读取choice delta mapping。"""
        raw_delta = choice.get("delta")
        if raw_delta is None:
            return {}
        delta = self._as_mapping(raw_delta)
        if not delta and raw_delta not in ({}, None):
            raise IrisProviderStreamProtocolError("choice delta 不是可解析的mapping")
        return delta

    def _consume_delta(self, delta: Mapping[str, Any]) -> list[ModelStreamEvent]:
        """按固定语义顺序消费thinking、text与tool-call fragments。"""
        events: list[ModelStreamEvent] = []
        reasoning = delta.get("reasoning_content", delta.get("reasoning"))
        if reasoning is not None:
            if not isinstance(reasoning, str):
                raise IrisProviderStreamProtocolError("reasoning delta 必须是字符串")
            if reasoning:
                events.extend(self._append_text_delta("thinking", reasoning))

        content = delta.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise IrisProviderStreamProtocolError("content delta 必须是字符串")
            if content:
                events.extend(self._append_text_delta("text", content))

        raw_tool_calls = delta.get("tool_calls")
        if raw_tool_calls is not None:
            if not isinstance(raw_tool_calls, list):
                raise IrisProviderStreamProtocolError("tool_calls delta 必须是list")
            for raw_tool_call in raw_tool_calls:
                events.extend(self._consume_tool_delta(self._as_mapping(raw_tool_call)))
        return events

    def _append_text_delta(
        self,
        kind: str,
        delta: str,
    ) -> list[ModelStreamEvent]:
        """追加text或thinking channel并返回对应events。"""
        key = (kind, 0)
        state = self._blocks.get(key)
        events: list[ModelStreamEvent] = []
        if state is None:
            block = self._new_block(kind=kind)
            state = _BlockState(block=block)
            self._blocks[key] = state
            events.append(ModelBlockStarted(**self._event_fields(), block=block))
        if state.completed:
            raise IrisProviderStreamProtocolError("completed block 后仍收到delta")
        state.snapshot += delta
        self._semantic_output_emitted = True
        events.append(
            ModelBlockDelta(
                **self._event_fields(),
                block=state.block,
                channel="thinking" if kind == "thinking" else "text",
                delta=delta,
                snapshot=state.snapshot,
            )
        )
        return events

    def _consume_tool_delta(
        self,
        tool_call: Mapping[str, Any],
    ) -> list[ModelStreamEvent]:
        """追加一个function tool-call fragment。"""
        if not tool_call:
            raise IrisProviderStreamProtocolError("tool call不是可解析的mapping")
        provider_index = tool_call.get("index")
        if isinstance(provider_index, bool) or not isinstance(provider_index, int):
            raise IrisProviderStreamProtocolError("tool call index必须是整数")
        if provider_index < 0:
            raise IrisProviderStreamProtocolError("tool call index不得为负数")
        tool_type = tool_call.get("type")
        if tool_type not in {None, "", "function"}:
            raise IrisProviderStreamProtocolError("只支持function tool call")

        key = ("tool_call", provider_index)
        state = self._blocks.get(key)
        events: list[ModelStreamEvent] = []
        raw_call_id = tool_call.get("id")
        if raw_call_id is not None and not isinstance(raw_call_id, str):
            raise IrisProviderStreamProtocolError("tool call id必须是字符串")
        if state is None:
            call_id = raw_call_id or (
                f"{self._scope.model_stream_id}:tool:{self._scope.attempt}:{provider_index}"
            )
            block = self._new_block(
                kind="tool_call",
                block_id=call_id,
                tool_call_id=call_id,
            )
            state = _BlockState(block=block)
            self._blocks[key] = state
            events.append(ModelBlockStarted(**self._event_fields(), block=block))
        elif raw_call_id and raw_call_id != state.block.tool_call_id:
            raise IrisProviderStreamProtocolError("tool call id在fragments之间不一致")
        if state.completed:
            raise IrisProviderStreamProtocolError("completed tool block后仍收到delta")

        function = self._as_mapping(tool_call.get("function") or {})
        name_delta = function.get("name")
        argument_delta = function.get("arguments")
        for field_name, value in (("name", name_delta), ("arguments", argument_delta)):
            if value is not None and not isinstance(value, str):
                raise IrisProviderStreamProtocolError(f"tool function {field_name}必须是字符串")
        if name_delta:
            state.tool_name += name_delta
            self._semantic_output_emitted = True
            events.append(
                ModelBlockDelta(
                    **self._event_fields(),
                    block=state.block,
                    channel="tool_name",
                    delta=name_delta,
                    snapshot=state.tool_name,
                )
            )
        if argument_delta:
            state.tool_arguments += argument_delta
            self._semantic_output_emitted = True
            events.append(
                ModelBlockDelta(
                    **self._event_fields(),
                    block=state.block,
                    channel="tool_arguments",
                    delta=argument_delta,
                    snapshot=state.tool_arguments,
                )
            )
        return events

    def _complete_blocks(self) -> list[ModelStreamEvent]:
        """按source order完成所有已开始的blocks。"""
        events: list[ModelStreamEvent] = []
        for state in sorted(self._blocks.values(), key=lambda item: item.block.index):
            if state.completed:
                continue
            if state.block.kind == "tool_call":
                self._validate_completed_tool(state)
            state.completed = True
            events.append(ModelBlockCompleted(**self._event_fields(), block=state.block))
        return events

    def _validate_completed_tool(self, state: _BlockState) -> None:
        """在block completion边界验证完整tool name与JSON object arguments。"""
        if not state.tool_name:
            raise IrisProviderStreamProtocolError("tool call缺少完整name")
        if not state.tool_arguments:
            raise IrisProviderStreamProtocolError("tool call缺少完整arguments")
        try:
            arguments = json.loads(state.tool_arguments)
        except json.JSONDecodeError as exc:
            raise IrisProviderStreamProtocolError("tool arguments不是合法JSON object") from exc
        if not isinstance(arguments, dict):
            raise IrisProviderStreamProtocolError("tool arguments必须是JSON object")
        state.parsed_arguments = arguments

    def _new_block(
        self,
        *,
        kind: str,
        block_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> ModelBlockRef:
        """按首次出现顺序分配稳定block identity。"""
        index = self._next_block_index
        self._next_block_index += 1
        stable_block_id = block_id or (
            f"{self._scope.model_stream_id}:block:{self._scope.attempt}:{index}"
        )
        return ModelBlockRef(
            index=index,
            block_id=stable_block_id,
            kind=kind,
            tool_call_id=tool_call_id,
        )

    def _build_response(self) -> LLMResponse:
        """从已完成的块构造响应，复用完成边界解析的工具参数。"""
        blocks = self._ordered_blocks()
        text = "".join(state.snapshot for state in blocks if state.block.kind == "text")
        reasoning = "".join(state.snapshot for state in blocks if state.block.kind == "thinking")
        content: list[ContentBlock] = [TextBlock(text=text)] if text else []
        content.extend(
            ToolUseBlock(
                id=cast(str, state.block.tool_call_id),
                name=state.tool_name,
                input=state.parsed_arguments,
            )
            for state in blocks
            if state.block.kind == "tool_call"
        )
        raw_object = self._raw_object.removesuffix(".chunk")
        return LLMResponse(
            provider=self._scope.provider,
            id=self._response_id_or_fallback(),
            model=self._model or self._scope.model,
            content=content,
            finish_reason=self._finish_reason,
            input_tokens=self._usage.input_tokens,
            output_tokens=self._usage.output_tokens,
            total_tokens=self._usage.total_tokens,
            reasoning=reasoning,
            metadata={"raw_object": raw_object or "chat.completion"},
        )

    def _ordered_blocks(self) -> list[_BlockState]:
        """返回按source order排列的blocks。"""
        return sorted(self._blocks.values(), key=lambda item: item.block.index)

    def _usage_from_mapping(
        self,
        usage: Mapping[str, Any],
        *,
        complete: bool,
    ) -> ModelUsageSnapshot:
        """把provider usage mapping转换为typed snapshot。"""
        try:
            return ModelUsageSnapshot(
                input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage.get("completion_tokens", 0) or 0),
                total_tokens=int(usage.get("total_tokens", 0) or 0),
                complete=complete,
            )
        except (TypeError, ValueError) as exc:
            raise IrisProviderStreamProtocolError("usage token字段无效") from exc

    def _response_id_or_fallback(self) -> str:
        """返回provider response id或attempt-local稳定fallback。"""
        return self._response_id or (
            f"{self._scope.model_stream_id}:response:{self._scope.attempt}"
        )

    def _event_fields(self) -> dict[str, Any]:
        """分配下一个连续provider sequence及aware UTC时间。"""
        fields: dict[str, Any] = {
            "scope": self._scope,
            "sequence": self._next_sequence,
            "occurred_at": datetime.now(UTC),
        }
        self._next_sequence += 1
        return fields

    # endregion


async def _iter_litellm_events(
    raw_stream: AsyncIterator[Any],
    *,
    scope: ModelStreamScope,
    as_mapping: _AsMapping,
    error_mapper: _ErrorMapper,
) -> AsyncIterator[ModelStreamEvent]:
    """直接拉取LiteLLM raw stream并只产出typed events。

    Args:
        raw_stream: LiteLLM返回的raw async iterator。
        scope: 当前provider attempt identity。
        as_mapping: LiteLLM object转换函数。
        error_mapper: LiteLLM异常到Iris provider异常的mapper。

    Yields:
        连续的provider-neutral模型流式事件。
    """
    accumulator = ModelStreamAccumulator(
        scope=scope,
        as_mapping=as_mapping,
    )
    try:
        try:
            async for raw_chunk in raw_stream:
                for event in accumulator.feed(raw_chunk):
                    yield event
            for event in accumulator.finish():
                yield event
        except IrisProviderStreamError as exc:
            yield accumulator.fail(exc)
        except Exception as exc:
            yield accumulator.fail(error_mapper(exc))
    finally:
        close = getattr(raw_stream, "aclose", None)
        if callable(close):
            try:
                await close()
            except Exception:
                _logger.warning("关闭provider raw stream失败", exc_info=True)


def failed_before_start(
    *,
    scope: ModelStreamScope,
    error: IrisProviderError,
    as_mapping: _AsMapping,
) -> ModelResponseFailed:
    """构造网络调用在首个chunk前失败时的唯一terminal。"""
    accumulator = ModelStreamAccumulator(
        scope=scope,
        as_mapping=as_mapping,
    )
    return accumulator.fail(error)


def _safe_provider_error(error: IrisProviderError) -> ProviderStreamError:
    """删除raw异常详情并保留稳定provider code。"""
    if isinstance(error, IrisProviderStreamInterruptedError):
        message = "provider stream在合法终态前结束"
    elif isinstance(error, IrisProviderStreamProtocolError):
        message = "provider stream响应协议无效"
    else:
        message = "provider stream调用失败"
    return ProviderStreamError(
        code=error.runtime_code,
        message=message,
        retryable=isinstance(
            error,
            (IrisAPIConnectionError, IrisRateLimitExceededError),
        ),
    )
