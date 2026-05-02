# Message 消息系统

为 agent、LLM 与工具之间的通信提供统一消息类型。所有交互（用户输入、LLM 回复、工具调用、工具结果）都通过同一个 `Msg` 类型流转，保证内存、序列化与 API 格式的一致性。

## Quick Start

```python
from iris.message import Msg, Conversation

# 创建单条消息
user_msg = Msg.user("修复 main.py 里的 bug")
assistant_msg = Msg.assistant("我来帮你修复。")
tool_result_msg = Msg.tool_result(
    tool_use_id="tool_abc", content="Bug 已修复", is_error=False
)

# 组织为会话
conv = Conversation()
conv.add(user_msg)
conv.add(assistant_msg)
conv.add(tool_result_msg)

# 转换为不同 LLM API 格式
openai_format = conv.to_openai()
anthropic_format = conv.to_anthropic()
```

## Important Definitions

### Role 角色枚举

- `Role.USER` - 用户消息
- `Role.ASSISTANT` - LLM 助手消息
- `Role.SYSTEM` - 系统提示消息
- `Role.TOOL` - 工具调用消息

### Content Blocks 内容块

消息内容可为单纯字符串，也可为结构化内容块列表：

- **TextBlock** - 纯文本内容
  - `text: str` - 文本内容

- **ToolUseBlock** - LLM 发起的工具调用
  - `id: str` - 唯一标识（自动生成）
  - `name: str` - 工具名称
  - `input: dict[str, Any]` - 工具参数

- **ToolResultBlock** - 工具执行结果
  - `tool_use_id: str` - 关联的 ToolUseBlock ID
  - `content: str` - 执行输出
  - `is_error: bool` - 是否执行失败

## API

### Msg 类

通用消息单元。

**属性：**
- `role: Role` - 消息角色
- `content: str | list[ContentBlock]` - 消息内容
- `sender: str` - 发送方名称（可选）
- `timestamp: float` - Unix 时间戳（秒）
- `metadata: dict[str, Any]` - 任意元数据

**快捷属性：**
- `text: str` - 提取并拼接所有文本内容
- `tool_calls: list[ToolUseBlock]` - 提取所有工具调用块
- `tool_results: list[ToolResultBlock]` - 提取所有工具结果块
- `has_tool_calls: bool` - 是否包含工具调用
- `blocks: list[ContentBlock]` - 始终以内容块列表形式返回

**工厂方法：**

- `Msg.system(content: str, **kwargs) -> Msg` - 创建系统提示消息
- `Msg.user(content: str, *, sender: str = "user", **kwargs) -> Msg` - 创建用户消息
- `Msg.assistant(content: str | list[ContentBlock], *, sender: str = "assistant", **kwargs) -> Msg` - 创建助手消息
- `Msg.tool_result(*, tool_use_id: str, content: str = "", is_error: bool = False, **kwargs) -> Msg` - 创建工具结果消息

**API 转换方法：**

- `to_openai(api_style: str = "responses") -> list[dict[str, Any]]` - 转换为 OpenAI 格式
  - `api_style="responses"` 用于 OpenAI Responses API
  - `api_style="chat"` 用于 OpenAI Chat Completions API
  
- `to_anthropic() -> dict[str, Any]` - 转换为 Anthropic Messages API 格式
  - 仅支持非系统消息（系统消息应通过 `system` 参数传递）

**反序列化方法：**

- `Msg.from_openai(response: dict[str, Any], *, api_style: str = "responses") -> Msg` - 解析 OpenAI 格式响应
  - `api_style="responses"` 用于 OpenAI Responses API
  - `api_style="chat"` 用于 OpenAI Chat Completions API

- `Msg.from_anthropic(response: dict[str, Any]) -> Msg` - 解析 Anthropic API 响应

### Conversation 类

有序消息集合，用于构建、管理与序列化会话历史。

**属性：**
- `messages: list[Msg]` - 消息列表

**写入方法：**
- `add(msg: Msg) -> None` - 追加单条消息
- `add_many(msgs: Sequence[Msg]) -> None` - 一次性追加多条消息

**查询属性：**
- `system_prompt: str | None` - 返回第一条系统消息文本
- `non_system_messages: list[Msg]` - 除系统提示外的全部消息
- `last: Msg | None` - 最近一条消息
- `turn_count: int` - 用户消息数量（会话轮数）

**API 序列化方法：**
- `to_openai() -> list[dict[str, Any]]` - 整个会话转换为 OpenAI 格式
- `to_anthropic() -> dict[str, Any]` - 整个会话转换为 Anthropic 格式（含 `system` 和 `messages`）

**上下文管理方法：**
- `estimate_tokens(chars_per_token: int = 4) -> int` - 粗略 token 数估算（启发式）
- `slice_recent(n: int) -> list[Msg]` - 返回最近 `n` 条非系统消息
- `clear(keep_system: bool = True) -> None` - 清空消息（可选保留系统提示）

**协议方法：**
- `__len__() -> int` - 返回消息总数
- `__iter__()` - 返回消息迭代器

## Examples

### 创建和查询消息

```python
from iris.message import Msg, Role, ToolUseBlock

# 创建包含工具调用的助手消息
tool_call = ToolUseBlock(name="bash", input={"command": "ls -la"})
assistant_msg = Msg.assistant([tool_call, TextBlock(text="执行以下命令")])

# 查询消息内容
print(assistant_msg.text)  # "执行以下命令"
print(assistant_msg.has_tool_calls)  # True
for call in assistant_msg.tool_calls:
    print(call.name)  # "bash"
```

### 处理会话历史

```python
conv = Conversation()
conv.add(Msg.system("你是一个有帮助的助手。"))
conv.add(Msg.user("今天天气怎么样？"))
conv.add(Msg.assistant("我无法实时查询天气。"))

# 检查会话状态
print(len(conv))  # 3
print(conv.turn_count)  # 1 (仅计算用户消息)
print(conv.system_prompt)  # "你是一个有帮助的助手。"

# 获取最近消息
recent = conv.slice_recent(n=2)  # 最后两条非系统消息
```

### 转换为 LLM API 格式

```python
# 为 OpenAI 调用准备
openai_messages = conv.to_openai()
# [
#   {"role": "system", "content": "你是一个有帮助的助手。"},
#   {"role": "user", "content": "今天天气怎么样？"},
#   {"role": "assistant", "content": "我无法实时查询天气。"}
# ]

# 为 Anthropic 调用准备
anthropic_payload = conv.to_anthropic()
# {
#   "system": "你是一个有帮助的助手。",
#   "messages": [
#     {"role": "user", "content": {"type": "text", "text": "..."}},
#     {"role": "assistant", "content": [...]}
#   ]
# }
```

### 处理 LLM 响应

```python
# OpenAI 响应反序列化
openai_response = {
    "choices": [
        {
            "message": {
                "content": "这是回复",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {"name": "bash"},
                        "arguments": '{"command": "pwd"}',
                    }
                ],
            }
        }
    ],
    "model": "gpt-4",
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}
msg = Msg.from_openai_chat(openai_response)
# msg.text      → "这是回复"
# msg.tool_calls → [ToolUseBlock(...)]
# msg.metadata   → {"model": "gpt-4", ...}

# Anthropic 响应反序列化
anthropic_response = {
    "content": [
        {"type": "text", "text": "我会帮你"},
        {"type": "tool_use", "id": "tool_456", "name": "search", "input": {...}},
    ],
    "model": "claude-3",
    "usage": {"input_tokens": 20, "output_tokens": 10},
}
msg = Msg.from_anthropic(anthropic_response)
```
