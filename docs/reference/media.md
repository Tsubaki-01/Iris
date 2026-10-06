# 消息、模型与媒体 SDK 参考

这些接口用于模型集成、宿主输入和自定义 provider。需要完整 Agent 运行时优先使用 [AgentRunner](runtime.md)，而不是手工循环调用模型。图片与语音的完整操作见[媒体配方](../cookbook/media.md)。

## 消息与内容块

从 `iris.message` 导入：

| 对象 | 主要字段或接口 |
| --- | --- |
| `Msg` | `role` 必填；`content=""` 为字符串或内容块列表；`sender=""`、`timestamp`、`metadata={}` |
| `Role` | `SYSTEM`、`USER`、`ASSISTANT`、`TOOL` |
| `TextBlock` | `text: str`，type 为 `text` |
| `ImageBlock` | `original: ImageFileRef`、`model: ImageFileRef`、`name=None` |
| `ImageFileRef` | `path: Path`、`mime_type`、`width`、`height`，引用已保存文件 |
| `ToolUseBlock` | `name`、`input={}`、`id`；用于将结果对应到具体调用 |
| `ToolResultBlock` | `tool_use_id`、有序 `content: list[TextBlock \| ImageBlock]`、`is_error=False`、`name=""`、`metadata={}` |

`DataBlock` 是文字或图片；`ContentBlock` 还包括工具调用与结果。用户的 `AgentRunRequest.input` 接受字符串或 `list[DataBlock]`，不能把任意 provider payload 当成输入块。

常用工厂为 `Msg.system(content)`、`Msg.user(content, *, sender="user")`、`Msg.assistant(content, *, sender="assistant")` 和 `Msg.tool_result(...)`。`msg.text` 聚合文本，`msg.blocks` 始终返回块列表，`tool_calls` / `tool_results` 分别提取调用和结果。读取 `.text` 不等于保留了图片。

`Conversation(messages=...)` 提供 `add`、`add_many`、`last`、`system_prompt`、`non_system_messages` 和 `to_llm_request(model, **options)`。它是消息容器，不拥有 Session/Run 持久化和恢复。

## 单次模型请求

`LLMRequest` 字段：

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `model` | 必填 | 模型名 |
| `messages` | `[]` | 已选定的本次输入 |
| `tools` | `[]` | `ToolSpec(name, input_schema, description="", strict=False)` 列表 |
| `temperature`、`top_p`、`max_tokens` | `None` | 采样与输出选项 |
| `tool_choice` | `None` | `auto / none / required` 或 `{name: ...}` |
| `response_format` | `None` | `text / json_object` 或 `{name, schema, strict?}` |
| `stream` | `False` | 是否走流式接口 |
| `timeout` | `None` | 请求级时限 |
| `provider_options` | `{}` | 少量 provider 参数；当前客户端传递 `num_retries` |
| `metadata` | `{}` | Iris 请求元数据 |

`LLMRequest.from_conversation(conversation, model, **options)` 复制会话列表作为本次消息。`api_style` 不属于请求参数，也不能放入 `provider_options`。

`LLMResponse` 包含 `provider`、`id`、`model`、`content`、`finish_reason`、输入/输出/总 token 计数、`reasoning` 和 `metadata`。`to_msg()` 将完整结果投影为助手消息。

## Provider 工厂与扩展

从 `iris.providers` 导入：

```text
parse_model_route(model: str) -> ModelRoute
create_provider_client(model: str | ModelRoute, *, api_style="responses",
    api_key=None, base_url=None, timeout=None, headers=None) -> ProviderClient
```

`ModelRoute` 包含逻辑 `provider` 和剥离前缀后的 `model`。client 按[进程配置规则](configuration.md#进程配置)读取注册与凭据。

公开调用：

| 接口 | 契约 |
| --- | --- |
| `estimate_input_tokens(request) -> int` | 估算所选协议的完整输入，包含工具及输出格式，不发起生成 |
| `await complete(request) -> LLMResponse` | 要求 `request.stream=False` |
| `stream(request) -> AsyncIterator[ModelStreamEvent]` | 要求 `request.stream=True`，直接 `async for`，不先 await 建流 |

自定义 `CompletionProvider` 必须实现估算和 `complete`。若还实现独立的 `StreamingProvider.stream`，runtime 可以使用其 typed streaming 能力。将实例通过 `AgentRunner.from_config*(provider=...)` 注入，此时不创建真实 provider client。

非流式调用将认证、连接、限流等失败映射到 `iris.exceptions` 的 provider 异常。流式远端失败由 `ModelResponseFailed` 表达；不要忽略失败终态，只拼接 delta 就当作成功响应。

### 模型流式事件

事件公共字段为 `scope`、从 1 开始的 `sequence`、`occurred_at`。scope 包含 `model_stream_id`、provider、model 和 attempt。

| `kind` | typed 事件 | 额外数据 |
| --- | --- | --- |
| `response.started` | `ModelResponseStarted` | response_id |
| `block.started` | `ModelBlockStarted` | block 引用 |
| `block.delta` | `ModelBlockDelta` | block、channel、delta、snapshot |
| `block.completed` | `ModelBlockCompleted` | block 引用 |
| `usage.updated` | `ModelUsageUpdated` | usage 快照及 complete 标志 |
| `response.completed` | `ModelResponseCompleted` | 完整 LLMResponse、semantic_output_emitted |
| `response.failed` | `ModelResponseFailed` | ProviderStreamError、semantic_output_emitted |
| `response.cancelled` | `ModelResponseCancelled` | 可选 error、semantic_output_emitted |

channel 为 `text`、`thinking`、`tool_name` 或 `tool_arguments`。block 的 `index`、`block_id` 与 kind 共同标识响应里的内容块；工具块还带 `tool_call_id`。usage 缺失或不完整不能推导为实际零消耗。流式工具参数需要等完整响应解析后才能执行。

## 图片

```text
await runner.import_image(source: Path | bytes, *, session_id: str,
    name: str | None = None) -> ImageBlock
```

路径参数为 `Path`，相对路径以 runner workspace 解析。导入会保存文件，但不创建 Run，也不占用一次会话运行准入。提交图片时沿用同一个 session ID。

当前支持静态 PNG、JPEG、WebP，格式从内容识别。原始字节保留；模型版最长边上限为 2000 像素、编码字节上限为 `15 * 1024 * 1024 // 4`（3.75 MiB）。已有图片满足限制且不需方向修正时复用原副本；否则按固定缩放与编码策略准备模型版，有限处理后仍无法满足条件则抛出 `IrisImageError`。

`ImageBlock` 不携带 base64 或 provider URL。provider 在构造请求时读取 model 文件并编码；普通 `.text` 投影不会产生视觉输入。图片目录在 runner 关闭后保留，恢复与 fork 需要继续保留引用文件。

## 语音转录

`speech` 配置：

| 字段 | 默认 | 启用时要求 |
| --- | --- | --- |
| `enabled` | `false` | 严格布尔值 |
| `adapter` | `None` | `doubao_asr` 或 `dashscope_funasr` |
| `endpoint` | `None` | 非空连接地址 |
| `model` | `None` | 非空语音服务模型或资源标识 |

从 `iris.speech` 导入 `SpeechConfig`、`SpeechClient`、`SpeechAdapter`、`TranscriptionEvent`、`create_speech_client`：

```text
create_speech_client(config, *, api_key=None) -> SpeechClient | None
SpeechClient(adapter: SpeechAdapter)
SpeechClient.stream(audio: AsyncIterable[bytes]) -> AsyncIterator[TranscriptionEvent]
```

工厂在 disabled 时返回 `None`。启用时优先用显式语音 key，否则读取 `provider_api_keys[adapter]`，不会退回全局 `api_key` 或主模型 key。构造不连接服务，开始迭代时才处理音频。

输入是 16 kHz、16-bit little-endian、单声道原始 PCM，块非空且按 2 字节对齐，不含 WAV 文件头。`TranscriptionEvent(text, is_final)` 的 text 是整段当前全文；成功结束最后产生 final，失败或提前取消不能伪装成最终识别。

自定义 adapter 直接实现 `stream(audio)`，再构造 `SpeechClient(adapter)`。宿主拥有音频源和设备，提前结束时用 `aclosing` 关闭转录流和音频迭代器。

依据：[消息模型](../../src/iris/message/message.py)、[请求响应](../../src/iris/message/llm.py)、[模型事件](../../src/iris/message/streaming.py)、[provider 协议](../../src/iris/providers/protocols.py)、[图片策略](../../src/iris/utils/images.py)、[语音工厂](../../src/iris/speech/factory.py)。
