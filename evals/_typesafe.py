"""Jev 实验共用的 HTTP 请求与外部响应解析。"""

from typing import Any

import httpx2
from pydantic import BaseModel, ValidationError

from iris.exceptions import IrisProviderError


async def post_system_one[ResponseT: BaseModel](
    client: httpx2.AsyncClient,
    api_key: str,
    payload: dict[str, Any],
    response_model: type[ResponseT],
) -> ResponseT:
    """提交一次 System One 请求，在响应进入实验代码时解析一次。"""
    try:
        response = await client.post(
            "https://api.typesafe.ai/v1/systemone",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        return response_model.model_validate_json(response.content)
    except httpx2.HTTPStatusError as exc:
        raise IrisProviderError(f"TypeSafe HTTP {exc.response.status_code}") from exc
    except (httpx2.RequestError, ValidationError) as exc:
        raise IrisProviderError("TypeSafe 请求或响应解析失败") from exc
