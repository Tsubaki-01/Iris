# Provider API Wrapper

Iris 的 provider 调用链以 provider-neutral 模型为边界，当前 active path 通过
LiteLLM Chat Completion 执行：

```text
Conversation -> LLMRequest -> ProviderClient -> litellm.acompletion()
litellm.acompletion() -> ProviderClient -> LLMResponse -> Msg -> Conversation
```

## 核心边界

- `Msg` 是 Iris 内部最小消息单元，只表达 role、content、sender、timestamp、metadata。
- `Conversation` 管理有序消息历史，并通过 `to_llm_request()` 构建一次模型调用。
- `LLMRequest` 表达一次调用的 model、messages、tools、采样参数和 provider 选项。
- `ProviderClient` 是 Iris 对 LiteLLM Chat Completion 的薄封装，负责构造
  `litellm.acompletion()` kwargs、传入 API key/base URL/headers，并把响应和错误映射回
  Iris 类型。
- `LLMResponse` 是 provider-neutral 响应，通过 `to_msg()` 回到 Iris 内部消息系统。

## Provider 白名单

当前 `ProviderClient` 和 factory 只显式支持以下 provider：

- `openai`
- `anthropic`
- `deepseek`

`ProviderClient(provider=...)` 会校验 provider 白名单；`create_provider_client("provider/model")`
会复用同一份白名单，并按以下优先级解析 API key：

1. 显式 `api_key=...`
2. `IRIS_{PROVIDER}_API_KEY`，例如 `IRIS_DEEPSEEK_API_KEY`
3. `iris.init_config(api_key=...)` 中的通用 key

`http_client` 注入已从 active API 删除。生产路径不再暴露可注入的 `httpx.AsyncClient`，
直接传入 `http_client=` 或其他未知构造参数会被拒绝。

## Chat Completion 策略

Iris 当前只支持 LiteLLM Chat Completion active path。`ProviderClient.complete()` 会把
Iris 消息格式化成 OpenAI Chat messages 形状，并调用：

```python
await litellm.acompletion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "你好", "name": "user"}],
    api_key="...",
)
```

工具 schema 也统一使用 OpenAI Chat 形状，即使配置的 provider 是 Anthropic 或 DeepSeek。
这是 LiteLLM chat bridge 的 runtime 边界，不改变 `LLMRequest.tools` 的公共字段类型。

## Responses 暂不支持

本阶段不支持 OpenAI Responses API active path。调用方如果在
`LLMRequest.provider_options` 中传入 `api_style="responses"`，`ProviderClient.complete()`
会抛出 `IrisProviderError`。

## 错误映射

`ProviderClient.complete()` 将 LiteLLM 或 provider 风格错误映射为 Iris 自定义异常：

- `401` / `403` -> `IrisAuthenticationError`
- `408`、连接错误或 timeout -> `IrisAPIConnectionError`
- `429` -> `IrisRateLimitExceededError`
- 其他 provider 错误 -> `IrisProviderError`

`complete()` 仅支持非流式调用。传入 `stream=True` 会抛出 `IrisProviderError`，streaming
将在后续单独设计。
