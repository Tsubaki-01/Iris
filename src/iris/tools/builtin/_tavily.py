"""Tavily 请求生命周期与外部响应解析。"""

from __future__ import annotations

from typing import Any, Literal

import httpx2
from pydantic import BaseModel, ValidationError

from ...exceptions import IrisToolExecutionError


async def post_tavily[ResponseT: BaseModel](
    *,
    endpoint: Literal["search", "extract"],
    api_key: str,
    payload: dict[str, Any],
    response_model: type[ResponseT],
) -> ResponseT:
    """发送一次请求并将服务响应解析为可信模型。

    Args:
        endpoint: Tavily 接口名称。
        api_key: 显式传入的 Tavily 凭据。
        payload: 已由工具输入边界校验并映射的请求内容。
        response_model: 当前接口消费的响应字段模型。

    Returns:
        经过一次完整解析的 API 响应。

    Raises:
        IrisToolExecutionError: 请求、HTTP 状态或响应解析失败。
    """
    try:
        async with httpx2.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://api.tavily.com/{endpoint}",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            response.raise_for_status()
    except httpx2.HTTPStatusError as exc:
        raise IrisToolExecutionError(
            f"Tavily {endpoint} HTTP {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx2.RequestError as exc:
        raise IrisToolExecutionError(f"Tavily {endpoint} 请求失败: {exc}") from exc

    try:
        return response_model.model_validate_json(response.content)
    except ValidationError as exc:
        raise IrisToolExecutionError(f"Tavily {endpoint} 响应无法解析: {exc}") from exc
