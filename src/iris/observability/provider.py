"""保持 provider 能力和业务结果的内部模型观测包装。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from typing import cast

from opentelemetry import context
from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind

from ..exceptions import IrisProviderError
from ..message import (
    LLMRequest,
    LLMResponse,
    ModelResponseCancelled,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelResponseStarted,
    ModelStreamEvent,
    ModelUsageSnapshot,
    ModelUsageUpdated,
)
from ..providers.protocols import CompletionProvider, StreamingProvider, streaming_provider_for
from .service import Observability

logger = logging.getLogger(__name__)


def observe_provider(
    provider: CompletionProvider, observability: Observability
) -> CompletionProvider:
    """在启用时保留原 provider 的 complete-only 或 streaming 能力。"""
    if not observability.enabled:
        return provider
    streaming = streaming_provider_for(provider)
    if streaming is not None:
        return _StreamingProvider(provider, streaming, observability)
    return _CompletionProvider(provider, observability)


def _usage(
    observability: Observability,
    span: Span,
    value: LLMResponse | ModelUsageSnapshot | Mapping[str, int],
) -> None:
    """仅投影当前 typed 响应或失败字典明确给出的计数。"""
    if isinstance(value, Mapping):
        attributes = {
            f"gen_ai.usage.{name}": value[name]
            for name in ("input_tokens", "output_tokens")
            if name in value
        }
    else:
        attributes = {}
        if "input_tokens" in value.model_fields_set:
            attributes["gen_ai.usage.input_tokens"] = value.input_tokens
        if "output_tokens" in value.model_fields_set:
            attributes["gen_ai.usage.output_tokens"] = value.output_tokens
    observability.attributes(span, attributes)


def _response(
    observability: Observability,
    span: Span,
    response: LLMResponse,
    *,
    truncated_fields: tuple[str, ...],
) -> None:
    """投影 provider 已返回的身份与结束原因，不推断路由标签。"""
    attributes: dict[str, str | tuple[str, ...]] = {}
    if response.provider:
        attributes["gen_ai.provider.name"] = response.provider
    if response.id:
        attributes["gen_ai.response.id"] = response.id
    if response.model:
        attributes["gen_ai.response.model"] = response.model
    if response.finish_reason:
        attributes["gen_ai.response.finish_reasons"] = (response.finish_reason,)
    observability.attributes(span, attributes)
    observability.record_response(span, response, truncated_fields=truncated_fields)


class _CompletionProvider:
    """记录单次非流式请求；估算仍直接由原 provider 执行。"""

    def __init__(self, provider: CompletionProvider, observability: Observability) -> None:
        self._provider = provider
        self._observability = observability

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """不采集纯本地估算。"""
        return self._provider.estimate_input_tokens(request)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """记录一次真实调用，原样返回响应或传播业务异常与取消。"""
        obs = self._observability
        span = obs.start_span(
            f"chat {request.model}",
            kind=SpanKind.CLIENT,
            attributes={"gen_ai.operation.name": "chat", "gen_ai.request.model": request.model},
        )
        try:
            truncated_fields = obs.record_request(span, request)
            with obs.use_span(span):
                response = await self._provider.complete(request)
            _response(obs, span, response, truncated_fields=truncated_fields)
            _usage(obs, span, response)
            obs.attributes(span, {"iris.model.outcome": "completed"})
            return response
        except asyncio.CancelledError:
            obs.attributes(span, {"iris.model.outcome": "cancelled"})
            raise
        except Exception as error:
            obs.attributes(span, {"iris.model.outcome": "failed"})
            obs.error(span, error)
            if isinstance(error, IrisProviderError) and "usage" in error.context:
                _usage(obs, span, cast(Mapping[str, int], error.context["usage"]))
            raise
        finally:
            obs.end_span(span)


class _StreamingProvider(_CompletionProvider):
    """让流存续跨越 yield，但只在实际 provider 操作中激活模型 context。"""

    def __init__(
        self,
        provider: CompletionProvider,
        streaming: StreamingProvider,
        observability: Observability,
    ) -> None:
        super().__init__(provider, observability)
        self._streaming = streaming

    def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
        """同步捕获创建位置的 context，延迟到首次消费才调用 provider。"""
        return self._stream(request, context.get_current())

    async def _stream(
        self, request: LLMRequest, parent: Context
    ) -> AsyncIterator[ModelStreamEvent]:
        """原样转交事件并在结束时记录最后已知用量与实际终态。"""
        obs = self._observability
        span = obs.start_span(
            f"chat {request.model}",
            kind=SpanKind.CLIENT,
            context=parent,
            attributes={"gen_ai.operation.name": "chat", "gen_ai.request.model": request.model},
        )
        outcome: str | None = None
        usage: LLMResponse | ModelUsageSnapshot | Mapping[str, int] | None = None
        events: AsyncIterator[ModelStreamEvent] | None = None
        scope_recorded = False
        try:
            truncated_fields = obs.record_request(span, request)
            with obs.use_span(span, context=parent):
                events = self._streaming.stream(request)
            while True:
                try:
                    with obs.use_span(span, context=parent):
                        event = await anext(events)
                except StopAsyncIteration:
                    if outcome is None:
                        outcome = "failed"
                        obs.error(span, "Provider stream ended without a terminal event")
                    break
                if not scope_recorded:
                    obs.attributes(
                        span,
                        {
                            "gen_ai.provider.name": event.scope.provider,
                            "gen_ai.response.model": event.scope.model,
                        },
                    )
                    scope_recorded = True
                if isinstance(event, ModelResponseStarted):
                    obs.attributes(span, {"gen_ai.response.id": event.response_id})
                elif isinstance(event, ModelUsageUpdated):
                    usage = event.usage
                elif outcome is None:
                    if isinstance(event, ModelResponseCompleted):
                        outcome = "completed"
                        if usage is None:
                            usage = event.response
                        _response(obs, span, event.response, truncated_fields=truncated_fields)
                    elif isinstance(event, ModelResponseFailed):
                        outcome = "failed"
                        obs.error(span, event.error.message)
                    elif isinstance(event, ModelResponseCancelled):
                        outcome = "cancelled"
                yield event
        except asyncio.CancelledError:
            if outcome is None:
                outcome = "cancelled"
            raise
        except Exception as error:
            if outcome is None:
                outcome = "failed"
                obs.error(span, error)
            if isinstance(error, IrisProviderError) and "usage" in error.context:
                usage = cast(Mapping[str, int], error.context["usage"])
            raise
        finally:
            try:
                if events is not None:
                    close = getattr(events, "aclose", None)
                    if close is not None:
                        try:
                            with obs.use_span(span, context=parent):
                                await close()
                        except asyncio.CancelledError:
                            if outcome is None:
                                outcome = "cancelled"
                            raise
                        except Exception:
                            logger.warning("关闭 provider stream 失败", exc_info=True)
            finally:
                obs.attributes(span, {"iris.model.outcome": outcome or "abandoned"})
                if usage is not None:
                    _usage(obs, span, usage)
                obs.end_span(span)
