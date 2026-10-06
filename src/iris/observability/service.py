"""统一的 OTel 记录与上下文操作，不接管业务执行和权威状态。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import INVALID_SPAN, Span, SpanKind, Status, StatusCode, TracerProvider
from opentelemetry.util.types import AttributeValue

from ..exceptions import IrisCancellationRequestedError, IrisConfigError
from . import content
from .config import AgentObservabilityConfig, ObservabilityExportConfig

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

    from ..message import LLMRequest, LLMResponse
    from ..tools.base import ToolResult

_logger = logging.getLogger(__name__)
_ASSOCIATION = otel_context.create_key("iris.observability.association")


@contextmanager
def _attached(context: Context) -> Iterator[None]:
    """仅隔离上下文操作故障，业务体异常原样传播。"""
    token = None
    try:
        token = otel_context.attach(context)
    except Exception:
        _logger.exception("观测上下文绑定失败")
    try:
        yield
    finally:
        if token is not None:
            otel_context.detach(token)


class Observability:
    """构建期固定策略与标准 OTel 服务；共享使用者不拥有 SDK。"""

    def __init__(
        self,
        capture_config: AgentObservabilityConfig | None = None,
        tracer_provider: TracerProvider | None = None,
        *,
        owned_provider: SDKTracerProvider | None = None,
    ) -> None:
        self._config = capture_config if capture_config is not None else AgentObservabilityConfig()
        self._tracer = tracer_provider.get_tracer("iris") if tracer_provider else None
        self._owned_provider = owned_provider

    @property
    def enabled(self) -> bool:
        """返回固定的采集开关，区别于某次 span 的采样结果。"""
        return self._config.enabled

    @classmethod
    def from_config(
        cls,
        capture_config: AgentObservabilityConfig,
        export_config: ObservabilityExportConfig,
        *,
        tracer_provider: TracerProvider | None = None,
    ) -> Observability:
        """借用宿主 provider 或自建私有 SDK，不注册 OTel global。

        Raises:
            IrisConfigError: 启用时没有导出来源，或缺少可选 SDK。
        """
        if not capture_config.enabled:
            return cls(capture_config)
        if tracer_provider is not None:
            return cls(capture_config, tracer_provider)
        if not export_config.traces_endpoint:
            raise IrisConfigError("启用 observability 需要 traces_endpoint 或 tracer_provider")
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            raise IrisConfigError("启用 OTLP 导出需要安装 iris[observability]") from exc
        provider = SDKTracerProvider(
            resource=Resource.create({"service.name": export_config.service_name}),
            shutdown_on_exit=False,
        )
        try:
            exporter = OTLPSpanExporter(
                endpoint=export_config.traces_endpoint,
                headers=export_config.headers,
                timeout=export_config.timeout_seconds,
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            return cls(capture_config, provider, owned_provider=provider)
        except Exception as exc:
            provider.shutdown()
            raise IrisConfigError("observability 导出配置初始化失败") from exc

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        context: Context | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> Span:
        """开始标准 span；记录失败不阻断业务。"""
        if self._tracer is None:
            return INVALID_SPAN
        try:
            values = dict(self.association(context=context))
            values.update(attributes or {})
            return self._tracer.start_span(name, context=context, kind=kind, attributes=values)
        except Exception:
            _logger.exception("观测 span 创建失败")
            return INVALID_SPAN

    def end_span(self, span: Span) -> None:
        """结束已创建区间，不覆盖业务返回或异常。"""
        try:
            span.end()
        except Exception:
            _logger.exception("观测 span 结束失败")

    @contextmanager
    def use_span(self, span: Span, *, context: Context | None = None) -> Iterator[None]:
        """短暂激活 span；流式调用在 yield 前退出该区间。"""
        if not self.enabled or span is INVALID_SPAN:
            yield
            return
        with _attached(trace.set_span_in_context(span, context)):
            yield

    @contextmanager
    def scope(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> Iterator[Span]:
        """组合标准 span 操作；领域结果仍由调用 owner 提供。"""
        if not self.enabled:
            yield INVALID_SPAN
            return
        span = self.start_span(name, kind=kind, attributes=attributes)
        try:
            with self.use_span(span):
                try:
                    yield span
                except IrisCancellationRequestedError:
                    raise
                except Exception as exc:
                    self.error(span, exc)
                    raise
        finally:
            self.end_span(span)

    @contextmanager
    def bind(
        self,
        attributes: Mapping[str, AttributeValue],
        *,
        replace: bool = False,
    ) -> Iterator[None]:
        """在唯一 OTel 私有键内绑定不可变关联；activation 替换自身身份。"""
        if not self.enabled:
            yield
            return
        values = {} if replace else dict(self.association())
        values.update(attributes)
        with _attached(otel_context.set_value(_ASSOCIATION, MappingProxyType(values))):
            yield

    def association(self, *, context: Context | None = None) -> Mapping[str, AttributeValue]:
        """读取已有不可变关联，供 owner 识别当前区间，不另存执行状态。"""
        if not self.enabled:
            return {}
        return cast(
            Mapping[str, AttributeValue], otel_context.get_value(_ASSOCIATION, context) or {}
        )

    @contextmanager
    def detached(self) -> Iterator[None]:
        """清空延后工作的 OTel 上下文，保留其他 Python ContextVars。"""
        if not self.enabled:
            yield
            return
        with _attached(Context()):
            yield

    def attributes(self, span: Span, values: Mapping[str, AttributeValue]) -> None:
        """记录现有事实，不验证已类型化的业务值。"""
        try:
            if span.is_recording():
                span.set_attributes(values)
        except Exception:
            _logger.exception("观测属性记录失败")

    def event(self, span: Span, name: str, values: Mapping[str, AttributeValue]) -> None:
        """记录由领域 owner 提供的少量结果或决定事件。"""
        try:
            if span.is_recording():
                span.add_event(name, values)
        except Exception:
            _logger.exception("观测事件记录失败")

    def error(self, span: Span, error: BaseException | str) -> None:
        """标记真实失败；取消和正常等待由 owner 单独分类。"""
        try:
            if span.is_recording():
                span.set_status(Status(StatusCode.ERROR, str(error)))
        except Exception:
            _logger.exception("观测错误状态记录失败")

    def record_request(self, span: Span, request: LLMRequest) -> tuple[str, ...]:
        """投影 effective 请求，返回本次调用要保留的截断字段。"""
        try:
            if self._config.capture_content and span.is_recording():
                values = content.request_attributes(request, self._config.max_content_chars)
                self.attributes(span, values)
                return tuple(cast(list[str], values.get("iris.content.truncated_fields", [])))
        except Exception:
            _logger.exception("观测请求正文投影失败")
        return ()

    def record_response(
        self,
        span: Span,
        response: LLMResponse,
        *,
        truncated_fields: tuple[str, ...] = (),
    ) -> None:
        """通过正文开关后投影 typed 响应，不重拼流式增量。"""
        try:
            if self._config.capture_content and span.is_recording():
                values = content.response_attributes(response, self._config.max_content_chars)
                if truncated_fields:
                    values["iris.content.truncated_fields"] = [
                        *truncated_fields,
                        *cast(list[str], values.get("iris.content.truncated_fields", [])),
                    ]
                self.attributes(span, values)
        except Exception:
            _logger.exception("观测响应正文投影失败")

    def record_tool(
        self,
        span: Span,
        arguments: dict[str, Any],
        result: ToolResult | None = None,
    ) -> None:
        """在 gate 后读取参数和已有最终结果；等待或异常时不伪造输出。"""
        try:
            if self._config.capture_content and span.is_recording():
                self.attributes(
                    span,
                    content.tool_attributes(arguments, result, self._config.max_content_chars),
                )
        except Exception:
            _logger.exception("观测工具正文投影失败")

    def _shutdown(self) -> None:
        """关闭尚未移交或正常退出的自建 SDK；借用 provider 不动。"""
        provider, self._owned_provider = self._owned_provider, None
        if provider is not None:
            try:
                provider.shutdown()
            except Exception:
                _logger.exception("观测 SDK 关闭失败")

    async def aclose(self) -> None:
        """在线程执行自建 SDK 的同步排空与关闭。"""
        if self._owned_provider is not None:
            await asyncio.to_thread(self._shutdown)
