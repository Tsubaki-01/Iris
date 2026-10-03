# 流式语音转录示例

从本地 WAV 读取 PCM，按音频时长分块发送，打印每次完整文字快照。默认只调用语音服务；
追加 `--submit` 才将正常结束后的非空最终文字提交给 Agent，并等待完整 RunResult。
此示例属于宿主代码，不为 `iris chat` 增加录音入口。

## 准备配置

两份可加载的完整 Agent 配置使用相同主模型与系统提示，仅 speech 部分不同：

| 配置 | 语音服务 | 需要调整的内容 |
| --- | --- | --- |
| [doubao.yaml](doubao.yaml) | 豆包流式 ASR | 使用已开通的资源 ID 和豆包语音 API key |
| [dashscope.yaml](dashscope.yaml) | 阿里 Fun-ASR-Realtime | 将 `{WorkspaceId}` 替换为实际业务空间，使用账号地域的 endpoint 和百炼 key |

两份 YAML 的完整内容也可在[语音 SDK README](../../src/iris/speech/README.md#yaml-配置与开关)
同页比较。顶层 model 是主 Agent 模型，speech.model 是 ASR 模型或资源，互相独立。

在 `.env` 配置所选服务的专属 key；如果使用 `--submit`，再配置主 Agent 的 key：

```dotenv
IRIS_PROVIDER_API_KEYS__DOUBAO_ASR=豆包语音APIKey
IRIS_PROVIDER_API_KEYS__DASHSCOPE_FUNASR=阿里百炼APIKey
IRIS_PROVIDER_API_KEYS__DEEPSEEK=主Agent模型APIKey
```

脚本通过 `iris.init_config(env_file=...)` 加载指定文件，不会默认读取 `.env`，语音凭据也不
回退到通用聊天 key。不使用的语音服务无需填 key。

准备一段 16 kHz、16-bit、单声道 PCM WAV，例如 `sample.wav`。示例检查容器格式后只发送
frames，不发送 WAV 头；不负责录音、MP3 解码或重采样。音频路径相对运行时工作目录解析。

## 同一入口切换服务商

从仓库根目录运行：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run python examples/audio/transcribe_stream.py --config examples/audio/doubao.yaml --audio sample.wav --env-file .env
uv run python examples/audio/transcribe_stream.py --config examples/audio/dashscope.yaml --audio sample.wav --env-file .env
```

每次显示的是当前全文，可能修正之前的识别。服务分句结束不会提前创建 Agent Run；
文件自然 EOF 后等待整次识别终态。两家识别结果不保证逐字相同。

将最终文字发送给 Agent：

```powershell
uv run python examples/audio/transcribe_stream.py --config examples/audio/doubao.yaml --audio sample.wav --env-file .env --submit --session-id voice-demo
```

此时脚本先完成并关闭 ASR 流，再构造 runner、等待 `runner.start(...)`，输出 RunResult JSON，
最后关闭自己拥有的 runner。识别失败、取消或空 final 不调用主模型；主模型失败不会重录或
重放音频，之前输出的转录文字仍可复用。

把所选 YAML 中的 `speech.enabled` 改为 false，即关闭语音路径：不打开音频文件、不构造
语音 adapter 或 runner。配置修改作用于新调用，不热切换正在进行的识别。

## SDK 接入

宿主负责文件/麦克风和音频源；Client 管理公共输入与委托关闭，adapter 管理原生连接和任务。
使用 async generator 时通过 `aclosing` 保证提前退出也释放资源。长期应用可保留最终文字供
编辑，然后交给已有 SessionManager，沿原有结果消费流程处理；无需采用这个示例的单 Run 宿主。

仓库测试使用短 WAV、内存 WebSocket 和模型替身验证真实装配路径，不需要音频设备或网络。
示例不保存原始音频，也不建立新的转录存储系统。
