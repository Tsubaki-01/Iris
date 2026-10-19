"""Web 服务共用的请求生命周期与外部响应解析。"""

from __future__ import annotations

from typing import Any, Literal

import httpx2
from pydantic import BaseModel, ValidationError

from ...exceptions import IrisToolExecutionError

WebProvider = Literal["tavily", "deepseek"]

_BASE_URLS = {
    "tavily": "https://api.tavily.com",
    "deepseek": "https://api.deepseek.com/v1/tools",
}
_LABELS = {"tavily": "Tavily", "deepseek": "DeepSeek"}


async def post_web[ResponseT: BaseModel](
    *,
    provider: WebProvider,
    endpoint: Literal["search", "extract", "fetch"],
    api_key: str,
    payload: dict[str, Any],
    response_model: type[ResponseT],
) -> ResponseT:
    """发送一次请求并将服务响应解析为可信模型。

    Args:
        provider: 构造工具时选定的 Web 服务。
        endpoint: 对应服务的接口名称。
        api_key: 显式传入的服务凭据。
        payload: 已由工具输入边界校验并映射的请求内容。
        response_model: 当前接口消费的响应字段模型。

    Returns:
        经过一次完整解析的 API 响应。

    Raises:
        IrisToolExecutionError: 请求、HTTP 状态或响应解析失败。
    """
    label = _LABELS[provider]
    try:
        async with httpx2.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{_BASE_URLS[provider]}/{endpoint}",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            response.raise_for_status()
    except httpx2.HTTPStatusError as exc:
        raise IrisToolExecutionError(
            f"{label} {endpoint} HTTP {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx2.RequestError as exc:
        raise IrisToolExecutionError(f"{label} {endpoint} 请求失败: {exc}") from exc

    try:
        return response_model.model_validate_json(response.content)
    except ValidationError as exc:
        raise IrisToolExecutionError(f"{label} {endpoint} 响应无法解析: {exc}") from exc
