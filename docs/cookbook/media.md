# 输入图片与处理语音

图片和语音有不同的接入路径：图片作为 typed 数据块进入模型请求；语音先由独立 ASR 服务转成文字，再由宿主决定是否提交给 Agent。两者都通过 Python SDK 使用，当前 `iris chat` 没有发图、录音或图片展示入口。

## 提交一张图片

先完成[Python SDK 入门](../getting-started/python-sdk.md)，并选择支持所用协议下视觉输入与工具调用的模型。把要分析的静态 PNG、JPEG 或 WebP 放在仓库根目录，命名为 `photo.png`。

在同一目录保存 `read_image.py`，使用已有 `agent.yaml`：

```python
"""导入一张图片并提交看图任务。"""

import asyncio
from pathlib import Path

from iris import init_config
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest
from iris.message import TextBlock


async def main() -> None:
    """图片导入与任务输入使用同一个 session。"""
    init_config()
    runner = AgentRunner.from_config_path("agent.yaml")
    try:
        image = await runner.import_image(
            Path("photo.png").resolve(), session_id="image-demo", name="示例图片"
        )
        result = await runner.start(
            AgentRunRequest(
                session_id="image-demo",
                input=[TextBlock(text="描述你实际看见的内容，模糊处请说明。"), image],
            )
        )
        print(result.run.phase.value, result.run.stop_reason)
        if result.assistant_message is not None:
            print(result.assistant_message.text)
        if result.error is not None:
            print(result.error.message)
    finally:
        await runner.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

```powershell
uv run python read_image.py
```

纯图片输入使用 `input=[image]`。导入函数保存原图和供模型使用的版本，再返回引用；你不需要在消息里手工拼接 base64。导入成功只能证明文件已经准备好，模型是否正确理解图片要检查实际回答。

也可运行仓库现有的[图片示例](../../examples/image/README.md)：

```powershell
uv run python -m examples.image.basic --image photo.png --session-id image-demo
```

该脚本使用旁边的 Agent 配置并打印 `RunResult`，可通过 `--config` 换成自己的视觉模型配置。

## 保留图片供后续查阅

图片缓存位于 workspace 的 `.iris/image-cache/` 下，按 session 分组。Runner 关闭不会删除这些文件。数据库中的消息保存图片引用，因此跨进程恢复或备份需要同时保留数据库与缓存文件。

会话 fork 仍可能引用源 session 的图片文件，不应只复制目标会话数据库内容后删除源缓存。历史被摘要后，摘要包含文字结论和图片引用，不能替代原图细节；配置 `file.read` 后，模型可以通过 `read_file` 回读相应图片路径。

格式、尺寸和引用结构见[图片参考](../reference/media.md#图片)。

## 将音频转成文字

当前提供豆包 ASR 与阿里 Fun-ASR 两种 adapter。它们使用独立凭据，不共享主 Agent 模型的 key。以豆包示例为例，在 PowerShell 中设置：

```powershell
$env:IRIS_PROVIDER_API_KEYS__DOUBAO_ASR = "替换为语音服务专属 key"
```

准备一个 PCM WAV 文件 `sample.wav`：16 kHz、16-bit little-endian、单声道。在仓库根目录运行：

```powershell
uv run python examples/audio/transcribe_stream.py --config examples/audio/doubao.yaml --audio sample.wav
```

脚本按音频时长发送 PCM 分块，并输出“当前转录”和“最终转录”。每个事件是当前整段全文，应替换界面文字，而不是把快照不断追加。只有正常完成且 `is_final=True` 的文字才是这次识别的最终结果。

默认只转录，不调用 Agent。若已配置主模型凭据，需要让 Agent 处理最终文字，可追加：

```powershell
uv run python examples/audio/transcribe_stream.py --config examples/audio/doubao.yaml --audio sample.wav --submit
```

切换阿里服务时使用 `examples/audio/dashscope.yaml`，先把其中的 `{WorkspaceId}` 替换为实际业务空间，并设置 `IRIS_PROVIDER_API_KEYS__DASHSCOPE_FUNASR`。服务 endpoint、model 和账号开通条件须与你的实际服务一致。

## 宿主需要负责什么

Iris 接收原始 PCM 流，不负责麦克风设备、WAV 解封装、重采样或音频转码；仓库示例负责读取符合要求的 WAV。交互宿主还应决定识别期间怎样展示文本、何时提交，以及退出时关闭音频来源。

`create_speech_client(config.speech)` 在关闭语音时返回 `None`。启用时返回的 `SpeechClient.stream(audio)` 是异步迭代器，提前退出应以 `contextlib.aclosing` 关闭它；[现有完整示例](../../examples/audio/transcribe_stream.py)展示了文件流和转录流的嵌套关闭。

这条路径没有文本转语音，也没有将主模型改成实时语音对话模型。原理见[消息、协议与媒体](../design/messages-media.md)，接口见[媒体参考](../reference/media.md)。
