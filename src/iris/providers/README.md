# iris.providers

`iris.providers` 是 Iris 框架的底层模型 API 交互模块。该模块将内部的统一数据结构请求通过 LiteLLM Chat Completion bridge 发送至对应的 LLM 厂商，并将返回内容解析为 Iris 的标准响应模型。

## Quick Start

以下示例展示了如何使用 `ProviderClient` 发送 `LLMRequest`：

```python
import asyncio
from iris.message import Msg
from iris.message.llm import LLMRequest
from iris.providers import ProviderClient

async def main():
    # 1. 实例化所需要的 Provider Client
    client = ProviderClient(provider="openai", api_key="your-api-key")

    # 2. 构造通用的 LLMRequest（模型定义位于 iris.message.llm 中）
    request = LLMRequest(model="gpt-4o", messages=[Msg.user("你好")])

    # 3. 发送请求并获取 LLMResponse
    response = await client.complete(request)
    print(response.to_msg().text)

if __name__ == "__main__":
    asyncio.run(main())
```

## Important Definitions

- **Provider 隔离原则**: 该包下的所有类仅处理通信层与协议内容构造。LLM 的公共模型（如 `LLMRequest` 和 `LLMResponse`）与消息类型（如 `Msg`, `Role`, `TextBlock`）均由外部 `iris.message.llm` 输入和构造，不在此模块定义。

## API

### `class ProviderClient`
Provider Chat Completion 调用层的实体。只负责将 `LLMRequest` 转换为 LiteLLM chat kwargs，并把 LiteLLM 响应和错误映射回 `LLMResponse` 与 `IrisProviderError` 等 Iris 自定义异常。

- **构造参数:**
  - `provider: str`: Provider 名称，目前 `ProviderClient` / factory 白名单为 `"openai"`、`"anthropic"`、`"deepseek"`。
  - `api_key: str`: 厂商鉴权所需的 API Key。
  - `base_url: str | None = None`: 覆盖原本的 Base URL。
  - `timeout: float | None = None`: 请求超时时间。
  - `headers: dict[str, str]`: 透传给 LiteLLM 的额外 headers。

- **`async def complete(request: LLMRequest) -> LLMResponse`**
    发起非流式 Chat Completion 请求。包含完整的报错映射流程。不支持 `stream=True` 与 Responses API 风格。

`ProviderClient` 不暴露可注入的 HTTP client；测试和 SDK 调用方应通过 monkeypatch
`litellm.acompletion()` 或注入 runtime provider 边界来控制调用行为。
构造器会拒绝未知参数，避免不受支持的输入被静默忽略。

### `ModelRoute` / `parse_model_route()` / `create_provider_client()`
Provider factory 负责把高层 `provider/model` 路由转换为具体 `ProviderClient`。

- `ModelRoute`: 保存 `provider` 与 provider 内部模型名。
- `parse_model_route(model: str) -> ModelRoute`: 解析 `openai/gpt-4o` 这类路由字符串。
- `create_provider_client(...) -> ProviderClient`: 校验显式 provider 白名单，解析 API key，并构造 `ProviderClient(provider=...)`。

Factory 只做 provider client 装配；新代码应从 `iris.providers` 导入这些对象。
