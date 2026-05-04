# Iris Message

为 agent、LLM 与工具之间的通信提供统一消息模型和厂商无关（provider-neutral）的调用边界定义。

## Quick Start

```python
from iris.message import Conversation, Msg

conv = Conversation()
conv.add(Msg.system("你是一个得力助手。"))
conv.add(Msg.user("你好"))

# 转换为供 provider 消费的请求对象
request = conv.to_llm_request(model="gpt-4o", temperature=0.7)
```

## Important Definitions

在 `iris.message` 内的核心内容块类型及枚举定义：

*   **`Role`**: 包含了 `USER`、`ASSISTANT`、`SYSTEM`、`TOOL` 四种常见角色的枚举。
*   **`ContentBlock`**: 等同于 `TextBlock | ToolUseBlock | ToolResultBlock` 的联合类型。
*   **`TextBlock`**: 包含简单的纯文本内容 (`text`)。
*   **`ToolUseBlock`**: LLM 发起的工具调用块，包含 `id`、`name`（工具名称）和 `input`（参数内容）。
*   **`ToolResultBlock`**: 工具执行的返回结果序列块，包含对应的 `tool_use_id`、`content` 和 `is_error` 报错标识。

## API

### `Msg`
表示一次通用且统一的消息单元（如用户输入、LLM 回复或工具结果），在系统内保持格式一致性。推荐使用其内置的工厂方法创建：

*   `Msg.system(content: str, **kwargs)`: 创建系统提示消息。
*   `Msg.user(content: str, **kwargs)`: 创建用户消息。
*   `Msg.assistant(content: str | list[ContentBlock], **kwargs)`: 创建包含纯文本或内容块列表（如工具调用）的助手消息。
*   `Msg.tool_result(tool_use_id: str, content: str = "", is_error: bool = False, **kwargs)`: 创建给 LLM 返回的工具执行结果（按照主流 API 习惯其底层角色实为 `USER` 角色）。
*   `text` (property): 拼接并返回消息包含的所有有效文本内容。
*   `tool_calls` / `tool_results` (property): 便捷过滤并返回当前消息下的工具调用块或结果块列表。

### `Conversation`
组成会话的有序消息集合，负责会话管理与 API 请求的构造转换。

*   `add(msg: Msg)`: 会话末尾追加一条消息。
*   `add_many(msgs: Sequence[Msg])`: 一次性追加多条消息。
*   `system_prompt` (property): 提取并返回会话中的首条系统引导文本。
*   `non_system_messages` (property): 过滤掉系统提示的全部用户、助手消息清单。
*   `to_llm_request(model: str, **options)`: 将会话历史转换为一次厂商无关的调用请求 `LLMRequest`。

### `LLMRequest`
一次完整的 LLM 请求级上下文抽象对象（导入自 `iris.message.llm`，统一向外暴露）。

*   **主要属性**: `model`、`messages`（已发送的消息历史）、`temperature`、`max_tokens`、`tools`、`response_format` 等。
*   `system_prompt()` / `non_system_messages()`: 协助外部 Adapter 将 system 拆分的便捷方法（比如兼容 Anthropic 把 system 单独抽离顶层字段的场景）。
*   `from_conversation(conversation: Conversation, model: str, **options)`: 使用当前会话快照进行初始化构建。

### `LLMResponse`
Provider-neutral 返回模型抽象，负责标准化接受并解析厂商 Raw Response。

*   **重要属性**: `provider`、`content`（模型输出的正文/工具块集）、`input_tokens` 与 `output_tokens` 相关的 token 计数等。
*   `to_msg() -> Msg`: 将大模型响应安全转换为内部系统的助手消息。厂商特有字段（包括模型名、停止原因和用量开销等）会被自动折叠进 `Msg.metadata` 内部以便上层做日志观测而不污染消息结构。

## Examples

### 建立与回传包含工具调用的消息通信

```python
from iris.message import Msg, TextBlock, ToolUseBlock

# 1. 产生一条携带了具体工具调用信息（也可能有附加文本）的助手回复
tool_call = ToolUseBlock(id="call_abc123", name="search", input={"query": "Iris 框架文档"})
assistant_msg = Msg.assistant(content=[TextBlock(text="让我查一下相关的资料"), tool_call])
assert assistant_msg.has_tool_calls

# 2. 将工具执行完毕的结果再次组装并传回上下文中
result_msg = Msg.tool_result(tool_use_id=tool_call.id, content="查询成功：文档链接...", is_error=False)
```