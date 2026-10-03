# 语音转录 SDK

`iris.speech` 将宿主提供的 PCM 音频流交给转录 adapter，返回当前全文快照。
录音、格式转换、展示和向 Agent 发送文字由宿主负责；转录本身不创建 Run。

## 公共调用

`SpeechClient` 接受满足 `SpeechAdapter` 协议的对象。`stream(audio)` 直接返回
异步生成器，不需要先 await；使用 `aclosing`，确保中途退出也关闭本次识别。

```python
from collections.abc import AsyncIterable
from contextlib import aclosing

from iris.speech import SpeechAdapter, SpeechClient


async def transcribe(audio: AsyncIterable[bytes], adapter: SpeechAdapter) -> str:
    client = SpeechClient(adapter)
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

## 豆包流式识别

宿主可以直接组合豆包 adapter，凭据使用 Iris 已有配置；`.env` 中设置
`IRIS_PROVIDER_API_KEYS__DOUBAO_ASR`，并显式加载 dotenv：

```python
import iris

from iris.speech.adapters.doubao import DoubaoASRAdapter

iris.init_config(env_file=".env")
adapter = DoubaoASRAdapter(
    endpoint="wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
    model="volc.seedasr.sauc.duration",
    api_key=iris.get_config().provider_api_keys["doubao_asr"],
)
```

把这个 adapter 交给上面的 transcribe 函数即可。model 是已开通的豆包语音资源 ID，
不是主 Agent 的聊天模型。构造不联网，实际推进 stream 才建立本次 WebSocket。
当前固定使用全文结果、标点、数字书面化与二遍修正，不启用语义顺滑或额外改写模型。

豆包只在 adapter 内保留一块音频，以便将最后真实音频作为负序号末包发送；
100 ms 分块会增加约一块的等待。收到服务整段末响应后才产生 final，
句子的 definite 标志不代表用户已经结束输入。连接期限为 10 秒，正常末包发送后
等待整段终态的期限为 10 秒；partial 不重置该期限，录音期间的静音不触发它。

帧编码遵循当前[官方 Demo](https://portal.volccdn.com/obj/volcfe/cloud-universal-doc/upload_f9323339af7c7ce6d15678622eb77ccf.zip)。
Demo 的音频序列化标志与旧版文字协议存在差异，实际账号仍需联调确认；
SDK 不通过失败后改协议或重放音频来自动兼容。

## 阿里 Fun-ASR 流式识别

同一个 transcribe 函数可以改用阿里 adapter。`.env` 配置
`IRIS_PROVIDER_API_KEYS__DASHSCOPE_FUNASR`；初始化 Iris 配置后构造：

```python
from iris import get_config
from iris.speech.adapters.dashscope_funasr import DashScopeFunASRAdapter

adapter = DashScopeFunASRAdapter(
    endpoint="wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",
    model="fun-asr-realtime",
    api_key=get_config().provider_api_keys["dashscope_funasr"],
)
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
实际模型在半句停止时的末句确认行为仍需用真实账号联调；协议依据见
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
