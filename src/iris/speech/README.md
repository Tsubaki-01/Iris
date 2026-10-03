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
