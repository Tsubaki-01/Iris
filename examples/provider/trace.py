"""使用进程内包装器记录一次 provider 调用。

Example:
    from examples.provider.trace import TracingProvider
"""

# region imports
from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from iris.config import init_config, is_config_initialized
from iris.message import LLMRequest, LLMResponse
from iris.providers import create_provider_client, parse_model_route
from iris.runtime import RuntimeProvider

from .basic import build_request

# endregion


@dataclass(slots=True)
class TraceRecord:
    """保存一次 provider 调用的请求、响应或错误。

    Attributes:
        request (LLMRequest): 发给委托 provider 的请求。
        response (LLMResponse | None): 成功时的标准化响应。
        error (str | None): 失败时的异常类型和消息。

    Example:
        record = TraceRecord(request=LLMRequest(model="fake-model"))
        record.snapshot()["response"]
    """

    request: LLMRequest
    response: LLMResponse | None = None
    error: str | None = None

    def snapshot(self) -> dict[str, object]:
        """返回可 JSON 序列化的 trace 快照。

        Returns:
            dict[str, object]: 包含请求、可选响应和可选错误的快照。
        """
        return {
            "request": self.request.model_dump(mode="json"),
            "response": (
                self.response.model_dump(mode="json") if self.response is not None else None
            ),
            "error": self.error,
        }


class TracingProvider:
    """记录调用后委托给另一个 RuntimeProvider。

    Attributes:
        delegate (RuntimeProvider): 实际执行调用的 provider。
        records (list[TraceRecord]): 按调用顺序保存的进程内记录。

    Example:
        provider = TracingProvider(delegate=client)
        response = await provider.complete(request)
    """

    def __init__(self, delegate: RuntimeProvider) -> None:
        """创建包裹指定 provider 的记录器。

        Args:
            delegate (RuntimeProvider): 实际完成请求的 provider。
        """
        self.delegate = delegate
        self.records: list[TraceRecord] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """记录一次调用结果后返回委托响应。

        Args:
            request (LLMRequest): 待发送的 provider-neutral 请求。

        Returns:
            LLMResponse: 委托 provider 返回的标准化响应。

        Raises:
            Exception: 委托 provider 调用失败时原样传播。
        """
        record = TraceRecord(request=request)
        self.records.append(record)
        try:
            response = await self.delegate.complete(request)
        except Exception as exc:
            record.error = f"{exc.__class__.__name__}: {exc}"
            raise
        record.response = response
        return response


def main(argv: Sequence[str] | None = None) -> int:
    """运行一次带进程内追踪的 provider 调用。

    Args:
        argv (Sequence[str] | None): 可选命令行参数；省略时读取当前进程参数。

    Returns:
        int: 成功时返回零。

    Raises:
        SystemExit: 命令行参数不合法或请求帮助时抛出。
        Exception: 配置或 provider 调用失败时原样传播。
    """
    parser = argparse.ArgumentParser(description="追踪 Iris provider 调用。")
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--prompt", default="用一句话介绍 Iris。")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args(argv)

    route = parse_model_route(args.model)
    env_file: Path | None = args.env_file
    if not is_config_initialized():
        init_config(env_file=str(env_file) if env_file is not None else None)
    provider = TracingProvider(create_provider_client(route))
    request = build_request(model=route.model, prompt=args.prompt)
    response = asyncio.run(provider.complete(request))
    print(response.to_msg().text)
    trace = [record.snapshot() for record in provider.records]
    print(json.dumps(trace, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
