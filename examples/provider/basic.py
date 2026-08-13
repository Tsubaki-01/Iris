"""使用当前 provider 工厂完成一次独立请求。

Example:
    from examples.provider.basic import build_request

    request = build_request(model="deepseek-chat", prompt="介绍 Iris")
"""

# region imports
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from iris.config import init_config, is_config_initialized
from iris.message import Conversation, LLMRequest, LLMResponse, Msg
from iris.providers import create_provider_client, parse_model_route
from iris.runtime import RuntimeProvider

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


async def complete_once(
    provider: RuntimeProvider,
    request: LLMRequest,
) -> LLMResponse:
    """使用注入的 provider 完成一次请求。

    Args:
        provider (RuntimeProvider): 提供完成调用的运行时 provider。
        request (LLMRequest): 待发送的 provider-neutral 请求。

    Returns:
        LLMResponse: provider 返回的标准化响应。

    Raises:
        Exception: 委托 provider 调用失败时原样传播。
    """
    return await provider.complete(request)


def main(argv: Sequence[str] | None = None) -> int:
    """运行一次基础 provider 调用。

    Args:
        argv (Sequence[str] | None): 可选命令行参数；省略时读取当前进程参数。

    Returns:
        int: 成功时返回零。

    Raises:
        SystemExit: 命令行参数不合法或请求帮助时抛出。
        Exception: 配置或 provider 调用失败时原样传播。
    """
    parser = argparse.ArgumentParser(description="调用 Iris provider。")
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
    response = asyncio.run(complete_once(provider, request))
    print(response.to_msg().text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
