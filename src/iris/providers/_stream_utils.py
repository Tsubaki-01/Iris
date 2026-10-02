"""两种 provider stream 共用的资源关闭和公开失败映射。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from ..exceptions import (
    IrisAPIConnectionError,
    IrisProviderError,
    IrisProviderStreamInterruptedError,
    IrisProviderStreamProtocolError,
    IrisRateLimitExceededError,
)
from ..message import ProviderStreamError

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
    """保留原生失败原因，沿用网络异常的公开错误文案。"""
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
