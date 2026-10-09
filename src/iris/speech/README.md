# 语音转录 SDK

`iris.speech` 将宿主提供的 PCM 音频流交给转录 adapter，返回当前全文快照。
录音、格式转换、展示和向 Agent 发送文字由宿主负责；转录本身不创建 Run。
范围是忠实转录：主 Agent 接收最终文字，当前模块不提供 TTS 或额外的语义改写。

## YAML 配置与开关

语音默认关闭。普通 Agent YAML 无需提供语音地址、模型或凭据；显式关闭也可以写：

```yaml
speech:
  enabled: false
```

以下两份均为完整 Agent 配置。顶层 model 是接收最终文字的主模型，speech.model 是
语音服务的模型或资源；切换语音服务不需要修改主模型、录音格式或事件消费者。

豆包，对应 [doubao.yaml](../../../examples/audio/doubao.yaml)：

```yaml
name: voice-agent
model: deepseek/deepseek-flash
system: |
  你是一个助手，根据用户提交的文字回答问题。
speech:
  enabled: true
  adapter: doubao_asr
  endpoint: wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async
  model: volc.seedasr.sauc.duration
```

阿里，对应 [dashscope.yaml](../../../examples/audio/dashscope.yaml)：

```yaml
name: voice-agent
model: deepseek/deepseek-flash
system: |
  你是一个助手，根据用户提交的文字回答问题。
speech:
  enabled: true
  adapter: dashscope_funasr
  endpoint: "wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"
  model: fun-asr-realtime
```

豆包 model 必须是账号已开通的语音资源 ID。阿里的 `{WorkspaceId}` 需要替换成实际
业务空间，endpoint 地域须与账号对应。两家原生协议不同，换服务商时同时选择
adapter、endpoint、model 和对应 key；相同协议的地址才可以仅换 endpoint/key。

凭据沿用 Iris 全局配置，不写入 YAML：

```dotenv
IRIS_PROVIDER_API_KEYS__DOUBAO_ASR=豆包语音APIKey
IRIS_PROVIDER_API_KEYS__DASHSCOPE_FUNASR=阿里百炼APIKey
IRIS_PROVIDER_API_KEYS__DEEPSEEK=主Agent模型APIKey
```

只需配置所选语音厂商的 key。仅转录不需要主模型 key，实际向 Agent 提交时才需要它。
应用初始化一次 `iris.init_config(env_file=".env")`；默认不会自动加载 dotenv。
工厂的显式 api_key 优先于对应专属 key；显式空 key 或缺专属 key 抛出 IrisConfigError，
不回退通用聊天 key。传入有效显式 key 时，可以不初始化全局配置。

`load_agent_config(path)` 只加载声明。宿主调用 `create_speech_client(config.speech)` 才装配
客户端；关闭时直接返回 None，不读取凭据、不创建 adapter、连接或任务。
构造开启的客户端也不联网，首次消费 stream 才连接。修改配置后创建新客户端，
当前识别由宿主显式取消，不热切换。

## 公共调用

`stream(audio)` 直接返回异步生成器，不需要先 await；使用 `aclosing`，确保中途退出
也关闭本次识别。下面的函数接收已加载的 `AgentConfig.speech`，两家使用同一段代码：

```python
from collections.abc import AsyncIterable
from contextlib import aclosing

from iris.speech import SpeechConfig, create_speech_client


async def transcribe(audio: AsyncIterable[bytes], speech: SpeechConfig) -> str:
    client = create_speech_client(speech)
    if client is None:
        return ""
    final_text = ""
    async with aclosing(client.stream(audio)) as events:
        async for event in events:
            print(event.text)  # 界面应覆盖预览，而不是追加。
            if event.is_final:
                final_text = event.text
    return final_text
```

仅在流正常退出后，将非空最终文字交给已有的 Agent 输入入口。失败或取消时不要提交预览。
客户端可以复用，每次 stream 的识别状态和资源由 adapter 独立创建。

已有 SessionManager 的宿主可将结果传给 `await manager.submit(text, mode="auto")`。
这是准入回执，后续结果继续使用宿主原有事件消费流程；auto 按真正提交时的状态选择新 Run
或 steer。转录函数不关闭借用的 manager，也不在录音开始时打断现有 Run。
有编辑界面时，将最终文字留在输入框再发送；提交失败可复用这份文字，无需重新转录。

完整可运行的 WAV 示例见 [examples/audio](../../../examples/audio/README.md)。同一脚本用
`--config` 选择两份 YAML，默认只转录；追加 `--submit` 才等待一次完整 Agent Run。

## 豆包流式识别

宿主可以直接组合豆包 adapter，凭据使用 Iris 已有配置；`.env` 中设置
`IRIS_PROVIDER_API_KEYS__DOUBAO_ASR`，并显式加载 dotenv：

```python
import iris

from iris.speech import SpeechClient
from iris.speech.adapters.doubao import DoubaoASRAdapter

iris.init_config(env_file=".env")
adapter = DoubaoASRAdapter(
    endpoint="wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
    model="volc.seedasr.sauc.duration",
    api_key=iris.get_config().provider_api_keys["doubao_asr"],
)
client = SpeechClient(adapter)
```

直接 Python 组合时使用 `SpeechClient(adapter).stream(audio)`。model 是已开通的豆包语音资源 ID，
不是主 Agent 的聊天模型。构造不联网，实际推进 stream 才建立本次 WebSocket。
当前固定使用全文结果、标点、数字书面化与二遍修正，不启用语义顺滑或额外改写模型。

豆包只在 adapter 内保留一块音频，以便将最后真实音频作为负序号末包发送；
100 ms 分块会增加约一块的等待。收到服务整段末响应后才产生 final，
句子的 definite 标志不代表用户已经结束输入。连接期限为 10 秒，正常末包发送后
等待整段终态的期限为 10 秒；partial 不重置该期限，录音期间的静音不触发它。

帧编码遵循当前[官方 Demo](https://portal.volccdn.com/obj/volcfe/cloud-universal-doc/upload_f9323339af7c7ce6d15678622eb77ccf.zip)。

## 阿里 Fun-ASR 流式识别

直接 Python 组合也可以改用阿里 adapter。`.env` 配置
`IRIS_PROVIDER_API_KEYS__DASHSCOPE_FUNASR`；初始化 Iris 配置后构造：

```python
from iris import get_config
from iris.speech import SpeechClient
from iris.speech.adapters.dashscope_funasr import DashScopeFunASRAdapter

adapter = DashScopeFunASRAdapter(
    endpoint="wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",
    model="fun-asr-realtime",
    api_key=get_config().provider_api_keys["dashscope_funasr"],
)
client = SpeechClient(adapter)
```

将 `{WorkspaceId}` 替换为实际业务空间，并使用账号地域对应的完整地址。
本 adapter 支持 Fun-ASR-Realtime task 协议，不能直接用 Paraformer 或 Qwen Realtime 模型替代。
跨服务商切换除了 endpoint/key，也要选择相应 adapter 和 model；宿主的录音与事件消费不变。

adapter 等 task-started 后发送原始 PCM，录音自然结束后发送 finish-task，
继续等待 task-finished。按 sentence_id 覆盖当前句，多个非空句子用换行连接；
sentence_end 只确认当前句。最终文字只包含已确认句子，未确认预览不自动升级为最终结果。

固定启用应用层 heartbeat，支持持续输入静音；服务的心跳结果不进入转录文字。
这不替代正常提供音频，也不意味着自动重连。连接、task-started 与发送 finish-task 后的
整任务终态分别使用 10 秒期限，后者不会随 partial 重置。
协议依据见
[客户端事件](https://help.aliyun.com/zh/model-studio/fun-asr-client-events)和
[服务端事件](https://help.aliyun.com/zh/model-studio/fun-asr-server-events)。

## 音频与结果

- 输入为 16 kHz、16-bit signed little-endian、单声道 raw PCM。
  推荐按采集节奏提供 100 ms（3200 bytes）一块，尾块可以更短；WAV 等容器头不属于 PCM。
- Client 在消费时检查每块非空、字节数按 16-bit 对齐；空流或非法首块不会启动 adapter。
  宿主保证采样率和声道，SDK 不重采样、不解码文件、不采集麦克风。
- `TranscriptionEvent(text, is_final)` 是冻结数据对象。text 是可修正的当前全文，
  相同文字的 final 仍是一次独立完成通知。
- 成功识别恰好有一个 final；它表示音频自然结束且服务确认整次识别完成。
  停顿、分句结束和 WebSocket 关闭本身不代表成功。空 final 表示没有可提交文字。

## Adapter 与资源归属

`SpeechAdapter.stream(audio)` 返回 `AsyncGenerator[TranscriptionEvent, None]`。
静态 Protocol 声明为普通 def 返回异步生成器；实现可使用带 yield 的 async def。
Client 不检查服务商字段，也不负责网络或文本聚合。

adapter 消费已经检查的音频块，负责厂商请求/响应转换、全文归一化和连接任务清理；
不读取 AgentConfig 或全局配置，不调用 Agent。每次生成器关闭时，adapter 应停止并 await
自己的任务、释放连接。Client 会关闭它委托的 adapter 生成器。

宿主继续拥有音频源和录音设备；取消消费任务后需完成自身资源清理。
裸 async for 的 break 不保证异步生成器已关闭，应退出示例中的 aclosing 作用域。

公共音频格式错误使用 `IrisSpeechError`；内置 adapter 的服务/协议错误也归入该类型，
沿用 provider 来源与 `SPEECH_ERROR`。宿主音频源异常和 `CancelledError` 保持原语义。
这些发生在普通文字提交之前，不自动成为 Agent 的 Run 错误。

使用与设计：[图片与语音](../../../docs/cookbook/media.md) · [消息与媒体参考](../../../docs/reference/media.md)。
