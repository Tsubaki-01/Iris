# iris.providers

`iris.providers` 是 Iris 框架的底层模型 API 交互模块。该模块将厂商格式转换（Adapter）与 HTTP 网络请求封装（Client）解耦，并将内部的统一数据结构请求发送至对应的 LLM 厂商并解析返回内容。该设计避免了模型定义的重复边界与直接与协议的强耦合。

## Quick Start

以下示例展示了如何使用 `ProviderClient` 配置 `OpenAIMessageAdapter` 发送 `LLMRequest`：

```python
import asyncio
from iris.message import Msg
from iris.message.llm import LLMRequest
from iris.providers import OpenAIMessageAdapter, ProviderClient

async def main():
    # 1. 实例化所需要的厂商 Adapter 与 Client
    adapter = OpenAIMessageAdapter()
    client = ProviderClient(adapter=adapter, api_key="your-api-key")

    try:
        # 2. 构造通用的 LLMRequest（模型定义位于 iris.message.llm 中）
        request = LLMRequest(model="gpt-4o", messages=[Msg.user("你好")])

        # 3. 发送请求并获取 LLMResponse
        response = await client.complete(request)
        print(response.to_msg().text)
    finally:
        # 4. 释放 HTTP 客户端资源
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Important Definitions

- **Provider 隔离原则**: 该包下的所有类仅处理通信层与协议内容构造。LLM 的公共模型（如 `LLMRequest` 和 `LLMResponse`）与消息类型（如 `Msg`, `Role`, `TextBlock`）均由外部 `iris.message.llm` 输入和构造，不在此模块定义。

## API

### `class ProviderAdapter`
Provider 数据格式适配器的抽象基类。只做纯数据格式层面的转换。

- `provider: str`: 目标 Provider 的名称标志。
- `default_api_style: str | None`: 默认的目标 API 风格区分标志。
- `to_provider_request(request: LLMRequest) -> dict[str, Any]`: 将通用的 `LLMRequest` 转换为当前 Provider 要求的 Payload 格式。
- `from_provider_response(response: Mapping[str, Any]) -> LLMResponse`: 将 Provider 返回的 JSON Dictionary 反序列化并转换为标准的 `LLMResponse`。

### `class OpenAIMessageAdapter(ProviderAdapter)`
用于对接 OpenAI API 的适配器实现。
- 默认 `provider` = `"openai"`
- 默认 `default_api_style` = `"chat"` （Chat Completions API）。支持通过 `provider_options["api_style"] = "responses"` 切换风格。

### `class AnthropicMessageAdapter(ProviderAdapter)`
用于对接 Anthropic Messages API 的适配器实现。自动将 system prompt 扁平化作为顶层字段发送。
- 默认 `provider` = `"anthropic"`

### `class ProviderClient`
Provider 真实 HTTP 调用层的实体。只负责传输、Endpoint 推导、Headers 注入、重试和网络层/服务端的错误处理及映射（映射为 `IrisProviderError` 等自定义异常）。

- **构造参数:**
  - `adapter: ProviderAdapter`: 指定使用哪种协议格式的适配器。
  - `api_key: str`: 厂商鉴权所需的 API Key。
  - `base_url: str | None = None`: 覆盖原本的 Base URL。
  - `timeout: float | None = None`: 请求超时时间。

- **`async def complete(request: LLMRequest) -> LLMResponse`**
    发起非流式的 HTTP 模型请求。包含完整的报错映射流程。不支持 `stream=True`。
- **`async def close() -> None`**
    安全释放内部的 `httpx.AsyncClient` 连接池。
