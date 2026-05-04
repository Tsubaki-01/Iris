# Provider API Wrapper

Iris 的 provider 调用链分为六层：

```text
Conversation -> LLMRequest -> ProviderAdapter -> ProviderClient -> 厂商 API
厂商 API -> ProviderClient -> ProviderAdapter -> LLMResponse -> Msg -> Conversation
```

## 核心边界

- `Msg` 是 Iris 内部最小消息单元，只表达 role、content、sender、timestamp、metadata。
- `Conversation` 管理有序消息历史，并通过 `to_llm_request()` 构建一次模型调用。
- `LLMRequest` 表达一次调用的 model、messages、tools、采样参数和 provider 选项。
- `ProviderAdapter` 只做格式转换，不读取 API key，不发 HTTP。
- `ProviderClient` 负责 HTTP 调用、鉴权 header、endpoint 选择和 provider 错误映射。
- `LLMResponse` 是 provider-neutral 响应，通过 `to_msg()` 回到 Iris 内部消息系统。

## OpenAI 默认策略

Iris 默认使用 OpenAI Chat Completions。`Conversation` 不直接生成 OpenAI
payload，而是先生成 provider-neutral 的 `LLMRequest`：

```python
request = conversation.to_llm_request("gpt-4o")
payload = OpenAIMessageAdapter().to_provider_request(request)
```

这会生成 `/v1/chat/completions` payload。该选择是为了贴合 Iris 当前
`Conversation.messages` / `Msg.role` 架构，并与 Anthropic Messages API 保持接近。

如需使用 OpenAI Responses API，显式指定：

```python
request = conversation.to_llm_request(
    "gpt-4o",
    provider_options={"api_style": "responses"},
)
payload = OpenAIMessageAdapter().to_provider_request(request)
```

Responses API 的差异限制在 `OpenAIMessageAdapter` 和 `ProviderClient` 内，上层业务代码不应直接依赖 provider raw payload。

## 错误映射

`ProviderClient.complete()` 将 HTTP 层错误映射为 Iris 自定义异常：

- `401` / `403` -> `IrisAuthenticationError`
- `429` -> `IrisRateLimitExceededError`
- `5xx` -> `IrisProviderError`
- 连接错误或 timeout -> `IrisAPIConnectionError`

`complete()` 仅支持非流式调用。传入 `stream=True` 会抛出 `IrisProviderError`，streaming 将在后续单独设计。
