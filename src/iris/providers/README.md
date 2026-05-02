# Providers

`providers` 包负责在系统统一的标准消息格式 (`Msg`) 与特定大语言模型（LLM）提供商格式（如 OpenAI 和 Anthropic）之间进行转换。

## 快速开始

```python
from iris.providers import OpenAIMessageAdapter
from iris.message import Msg

adapter = OpenAIMessageAdapter(api_style="chat")

# 将标准消息转换为 OpenAI API 载荷
payload = adapter.to_provider(Msg.user("Hello"))

# 将 OpenAI API 响应解析回标准格式
llm_response = adapter.from_provider(api_return_dict)
msg = llm_response.to_msg()
```

## 重要定义

### `LLMResponse`
从原始模型 API 载荷解析而来的提供商中立的 LLM 响应对象。适配器在将其转换为统一的 `Msg` 对象之前，使用它作为结构化的中间格式。

**属性:**
- `content` (`list[ContentBlock]`): 解析后的内容块。
- `provider` (`str`): LLM 提供商名称（例如 `"openai"`, `"anthropic"`）。
- `id` (`str`): 响应的唯一标识符。
- `model` (`str`): 生成该响应的确切模型版本。
- `finish_reason` (`str`): 模型停止生成的原因（例如 `"stop"`, `"length"`）。
- `input_tokens` (`int`): 提示词使用的 Token 数量。
- `output_tokens` (`int`): 生成的 Token 数量。
- `total_tokens` (`int`): 消耗的 Token 总数。
- `reasoning` (`str`): 模型的推理过程内容（如果有）。
- `metadata` (`dict[str, Any]`): 特定于提供商的附加元数据。

**方法:**
- `to_msg() -> Msg`: 构建并返回一个结合了模型内容及相关元数据的统一助手 `Msg` 对象。

## API

### `MessageAdapter`
特定 LLM 提供商消息适配器的抽象基类。

**方法:**
- `to_provider(msg: Msg) -> Any`: 将统一的 `Msg` 转换为特定 LLM API 所需的原始格式。
- `from_provider(response: dict[str, Any]) -> LLMResponse`: 将原始 API 响应转换为 `LLMResponse` 对象。

### `OpenAIMessageAdapter(MessageAdapter)`
为 OpenAI API 专门序列化和解析消息。

**属性:**
- `api_style` (`Literal["responses", "chat"]`): 指定 API 的交互模式。默认为 `"chat"`。

**方法:**
- `to_provider(msg: Msg) -> list[dict[str, Any]]`: 根据 `api_style` 将转换逻辑分发给 `to_chat` 或 `to_responses`。
- `from_provider(response: dict[str, Any]) -> LLMResponse`: 根据 `api_style` 解析 OpenAI 的响应。

### `AnthropicMessageAdapter(MessageAdapter)`
将 Iris 标准消息序列化为 Anthropic 格式，并解析其返回值。

**方法:**
- `to_provider(msg: Msg) -> dict[str, Any]`: 将统一的 `Msg` 转换为 Anthropic 要求的数据字典。*如果 `msg` 包含 System 角色，则会引发 `ValueError`（系统提示应单独独立传递）。*
- `from_provider(response: dict[str, Any]) -> LLMResponse`: 将原始 Anthropic API 响应解析为标准的 `LLMResponse`。
