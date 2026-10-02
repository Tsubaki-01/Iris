# SDK 图片输入

[basic.py](basic.py) 通过 `AgentRunner.import_image()` 导入本地图片，再将文字和图片一起提交给
YAML 配置的模型。输入、回答和图片文件引用保存在 SQLite，会话结束后可以继续使用。
所有命令从仓库根目录执行。

## 运行

在 `.env.local` 中配置示例使用的逻辑 DeepSeek 凭据：

```dotenv
IRIS_PROVIDER_API_KEYS__DEEPSEEK=你的密钥
```

也可设置同名环境变量并省略 `--env-file`。凭据通过 `iris.config.init_config()` 加载。
真实模型调用会消耗服务额度。

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run python -m examples.image.basic --env-file .env.local --image "path/to/photo.png" --session-id image-demo --prompt "描述这张图片，并读出能看清的文字。"

# 空 prompt 提交纯图片输入。
uv run python -m examples.image.basic --env-file .env.local --image "path/to/photo.png" --session-id image-only --prompt=""
```

`--image` 必填，来源路径相对当前工作目录解析，支持静态 PNG、JPEG 和 WebP，格式按内容识别。
`--config` 默认使用旁边的 [agent.yaml](agent.yaml)。省略 `--session-id` 时创建随机会话；
复用同一 ID 可以继续已有会话，每次执行仍会导入并提交本次指定的图片。

示例最多执行 8 个模型步骤，输出 `RunResult` JSON：`assistant_message` 包含回答，
`run.stop_reason` 为 `completed` 时退出码为 0，其余运行状态为 1。导入失败时不会创建 run；
成功和失败路径都会关闭 runner。

## 模型与协议

默认配置为 `deepseek/deepseek-flash` 与 `model.api_style: responses`。
改成 `chat_completions` 后重建 Agent 即可选择 Chat Completions，SDK 输入无需变化。
主模型必须支持所选协议下的视觉输入与工具调用，不能假设每个 provider/model 都支持两种协议；
选择 Chat 不会让旧文字模型获得视觉能力。

DeepSeek 两种协议复用逻辑 DeepSeek 凭据。Responses 使用 LiteLLM `openai` 传输发送原生
`/responses`，Chat 使用 `deepseek` 传输发送 `/chat/completions`，失败不会自动切换协议。
Responses 工具图片放入对应 `call_id` 的原生 output；Chat adapter 保留工具回执，随后追加
带工具来源关联的 user 图片消息。该追加仅发生在发送请求时，不改变已存历史。
详见 [provider 图片契约](../../src/iris/providers/README.md#图片输入)。

## SDK 调用与历史回读

示例的核心顺序是先确定 session，再导入图片，最后提交同一 session 的请求：

```python
from pathlib import Path

from iris.config import init_config
from iris.harness import AgentRunner, AgentRunRequest, RunResult
from iris.message import TextBlock


async def describe_image() -> RunResult:
    """配置凭据并提交一张图片。"""
    init_config(env_file=".env.local")
    runner = AgentRunner.from_config_path(Path("examples/image/agent.yaml"))
    try:
        session_id = "image-demo"
        image = await runner.import_image(
            Path("path/to/photo.png").resolve(), session_id=session_id, name="photo.png"
        )
        return await runner.start(
            AgentRunRequest(
                input=[TextBlock(text="描述图片中的内容。"), image],
                session_id=session_id,
            )
        )
    finally:
        await runner.aclose()
```

`import_image()` 也接受图片 `bytes`；直接传相对 `Path` 时 SDK 按 runner workspace 解析，
因此示例先调用 `.resolve()`。纯图请求使用 `input=[image]`，后续文字追问可使用
`AgentRunRequest(input="刚才图片中有什么？", session_id=session_id)`。

文字摘要只保留已经表达的视觉结论和图片引用，不会自动保存全部视觉细节。需要细节时，
模型可先用 `context_read` 找到历史引用，再通过 `read_file` 读取 `model` 路径重新看图。
这要求配置 `file.read`，示例已启用；未开放该工具时由 host 重新提交图片。
普通看图使用 `model` 版，`original` 留给进一步处理。

`iris chat` 当前只提供文字输入与文字输出，没有发图或图片渲染界面；宿主通过 SDK 提交图片。
图片 token 预算是本地近似值，实际 usage 以服务端返回为准。

## 缓存与恢复

默认 YAML 的 workspace 为 `examples/image`，数据保存在：

- `.iris/image.db`：请求、会话历史和运行结果中的 typed 图片引用。
- `.iris/image-cache/<session目录>/`：`original` 原始字节与按需处理的 `model` 文件；
  无需处理时两者指向同一文件。

路径相对 YAML 所在目录解析，session ID 转换为可用目录名。导入保存的是独立副本，之后运行、
恢复及历史回读不再依赖来源文件。`runner.aclose()` 和 run 结束均保留缓存；fork 的旧图片继续
引用源 session 目录，新导入图片写入分支自己的目录，因此源缓存也是分支恢复所需的数据。

备份必须同时保存数据库与 `image-cache`，单独复制数据库不足以恢复图片。当前没有自动回收、
跨机器路径改写或旧工具正文 schema 双读机制。缓存不可读、导入失败或 provider 拒绝时沿
对应错误通道返回，不会静默去图或改成纯文字重试。

## 离线验证

[示例测试](../../tests/examples/test_image_example.py) 使用真实 YAML 装配、图片导入和 SQLite，
只替换模型响应；覆盖文字加图片、纯图片、关闭后缓存与历史保留，以及导入失败时关闭 runner。
这不代表服务端模型已经通过视觉验收。

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run pytest -q -p no:cacheprovider --basetemp="$PWD\tmp\pytest-image-example-$([guid]::NewGuid().ToString('N'))" tests/examples/test_image_example.py
```
