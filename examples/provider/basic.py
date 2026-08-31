"""使用当前 provider 工厂流式完成一次独立请求。

Example:
    from examples.provider.basic import build_request

    request = build_request(model="deepseek-chat", prompt="介绍 Iris")
"""

# region imports
from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from iris.config import init_config, is_config_initialized
from iris.exceptions import IrisProviderStreamError, IrisProviderStreamInterruptedError
from iris.message import (
    Conversation,
    LLMRequest,
    LLMResponse,
    ModelBlockDelta,
    ModelResponseCancelled,
    ModelResponseCompleted,
    ModelResponseFailed,
    Msg,
    ProviderStreamError,
)
from iris.providers import create_provider_client, parse_model_route
from iris.runtime import StreamingRuntimeProvider

# endregion


def build_request(*, model: str, prompt: str) -> LLMRequest:
    """构造一次 provider-neutral 请求。

    Args:
        model (str): 不带 provider 前缀的内部模型名。
        prompt (str): 本次请求的用户输入。

    Returns:
        LLMRequest: 包含固定系统提示和用户输入的请求。
    """
    conversation = Conversation(
        messages=[
            Msg.system("你是一个简洁的助手。"),
            Msg.user(prompt),
        ]
    )
    return conversation.to_llm_request(model, temperature=0.2)


async def stream_once(
    provider: StreamingRuntimeProvider,
    request: LLMRequest,
    *,
    output: TextIO,
) -> LLMResponse:
    """使用注入的 provider 增量写出文本并返回完整响应。

    Args:
        provider (StreamingRuntimeProvider): 提供流式调用的运行时 provider。
        request (LLMRequest): 待发送的 provider-neutral 请求。
        output (TextIO): 接收文本增量的输出流。

    Returns:
        LLMResponse: 成功终态携带的标准化完整响应。

    Raises:
        IrisProviderStreamError: Provider 返回失败或取消终态。
        IrisProviderStreamInterruptedError: 流在合法终态前结束。
    """
    response: LLMResponse | None = None
    stream_error: ProviderStreamError | None = None
    cancelled = False
    terminal_provider: str | None = None
    stream_request = request.model_copy(update={"stream": True})

    async for event in provider.stream(stream_request):
        if isinstance(event, ModelBlockDelta) and event.channel == "text":
            output.write(event.delta)
            output.flush()
        elif isinstance(event, ModelResponseCompleted):
            response = event.response
        elif isinstance(event, ModelResponseFailed):
            stream_error = event.error
            terminal_provider = event.scope.provider
        elif isinstance(event, ModelResponseCancelled):
            stream_error = event.error
            cancelled = True
            terminal_provider = event.scope.provider

    if response is not None:
        return response
    if stream_error is not None:
        raise IrisProviderStreamError(
            stream_error.message,
            provider=terminal_provider,
            code=stream_error.code,
            retryable=stream_error.retryable,
        )
    if cancelled:
        raise IrisProviderStreamError(
            "Provider 取消流式响应",
            provider=terminal_provider,
        )
    raise IrisProviderStreamInterruptedError("Provider stream 在合法终态前结束")


def main(argv: Sequence[str] | None = None) -> int:
    """运行一次基础 provider 流式调用。

    Args:
        argv (Sequence[str] | None): 可选命令行参数；省略时读取当前进程参数。

    Returns:
        int: 成功时返回零。

    Raises:
        SystemExit: 命令行参数不合法或请求帮助时抛出。
        Exception: 配置或 provider 调用失败时原样传播。
    """
    parser = argparse.ArgumentParser(description="流式调用 Iris provider。")
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--prompt", default="用一句话介绍 Iris。")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args(argv)

    route = parse_model_route(args.model)
    env_file: Path | None = args.env_file
    if not is_config_initialized():
        init_config(env_file=str(env_file) if env_file is not None else None)
    provider = create_provider_client(route)
    request = build_request(model=route.model, prompt=args.prompt)
    asyncio.run(stream_once(provider, request, output=sys.stdout))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
