"""两种协议共用的流关闭、失败投影与首事件前失败构造。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from ..exceptions import (
    IrisAPIConnectionError,
    IrisProviderError,
    IrisProviderStreamInterruptedError,
    IrisProviderStreamProtocolError,
    IrisRateLimitExceededError,
)
from ..message import ModelResponseFailed, ModelStreamScope, ProviderStreamError

_logger = logging.getLogger(__name__)


async def close_raw_stream(raw_stream: AsyncIterator[Any]) -> None:
    """关闭支持 aclose 的底层流，不覆盖当前取消或原始失败。"""
    close = getattr(raw_stream, "aclose", None)
    if callable(close):
        try:
            await close()
        except Exception:
            _logger.warning("关闭 provider raw stream 失败", exc_info=True)


def safe_provider_error(error: IrisProviderError) -> ProviderStreamError:
    """将两种协议的失败原因与网络异常投影为统一公开错误。"""
    if isinstance(error, IrisProviderStreamInterruptedError):
        message = "provider stream在合法终态前结束"
    elif isinstance(error, IrisProviderStreamProtocolError):
        message = "provider stream响应协议无效"
    elif error.context.get("status"):
        reason = error.context.get("reason") or error.message
        message = f"Provider {error.context['status']}: {reason}"
    else:
        message = "provider stream调用失败"
    return ProviderStreamError(
        code=error.runtime_code,
        message=message,
        retryable=isinstance(error, (IrisAPIConnectionError, IrisRateLimitExceededError)),
    )


def failed_before_start(
    *, scope: ModelStreamScope, error: IrisProviderError
) -> ModelResponseFailed:
    """在尚未收到协议事件时构造两种协议共用的失败终态。"""
    return ModelResponseFailed(
        scope=scope,
        sequence=1,
        occurred_at=datetime.now(UTC),
        error=safe_provider_error(error),
        semantic_output_emitted=False,
    )
