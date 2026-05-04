"""Provider API wrapper 基础示例。"""

from __future__ import annotations

import asyncio
import os

from iris.message import Conversation, Msg
from iris.providers import OpenAIMessageAdapter, ProviderClient


async def main() -> None:
    """构建会话并调用 OpenAI Chat Completions。"""
    api_key = os.environ["OPENAI_API_KEY"]
    conversation = Conversation(
        messages=[
            Msg.system("你是一个简洁的助手。"),
            Msg.user("用一句话介绍 Iris。"),
        ]
    )
    request = conversation.to_llm_request("gpt-4o", temperature=0.2)
    client = ProviderClient(adapter=OpenAIMessageAdapter(), api_key=api_key)

    try:
        response = await client.complete(request)
        conversation.add(response.to_msg())
        print(conversation.last.text if conversation.last else "")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
