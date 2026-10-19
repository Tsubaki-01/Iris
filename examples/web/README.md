# Web 搜索与网页读取

这些示例通过 Iris 的 `web_search` / `web_fetch` 调用真实 Tavily 或 DeepSeek API，支持独立工具调用和
真实聊天模型驱动的搜索、读取、引用回答。所有命令从仓库根目录执行。

## 配置

在 `.env.local` 中配置凭据，运行时用 `--env-file .env.local` 显式加载：

```dotenv
IRIS_PROVIDER_API_KEYS__TAVILY=tvly-你的密钥
IRIS_PROVIDER_API_KEYS__DEEPSEEK=你的密钥
```

独立工具调用只需要所选服务的 key。`agent.yaml` 使用 Tavily Web 工具和 DeepSeek 模型，
需要两种 key；`deepseek.yaml` 的模型和 Web 工具共用 DeepSeek key。
也可以直接设置同名环境变量并省略 `--env-file`。配置通过现有 `iris.config.init_config()` 加载。
真实调用会消耗对应服务额度。

## 独立工具调用

[tools.py](tools.py) 使用 `build_tool_registry()` 和 `ToolExecutor.execute_one()`，输出与模型
收到的相同 Markdown。搜索参数、错误处理和长结果 artifact 都经过真实工具执行链。
两个示例及正式 `iris chat` 命令入口均使用 UTF-8 输出，支持 Windows 管道中的网页 Unicode 字符。

### DeepSeek

使用 `--provider deepseek`，查询语法原样送到服务端，Fetch 每次读取一个 URL：

```powershell
uv run python -m examples.web.tools --env-file .env.local --provider deepseek search "Python asyncio TaskGroup site:docs.python.org" --max-results 3
uv run python -m examples.web.tools --env-file .env.local --provider deepseek fetch https://docs.python.org/3/library/asyncio-task.html
```

Search 的 `query` 连同操作符最多 128 个 Unicode 字符，`max_results` 默认为 10、上限为 50。
搜索返回标题、URL 和供模型阅读的 `summary`；使用 `site:`、`-site:`、`after:`、`before:`
等语法限定来源。Fetch 返回标题和完整 Markdown 正文，不接受 `--query` 或多个 URL。
完整规则见 [Web 参数参考](../../docs/reference/tools.md#内置工具目录)与
[DeepSeek 官方指南](https://api-docs.deepseek.com/zh-cn/guides/search/)。

### Tavily

省略 `--provider` 或设置 `--provider tavily`：

```powershell
# 搜索：包含域名、排除域名与结果数量。
uv run python -m examples.web.tools --env-file .env.local search "Python asyncio TaskGroup" --max-results 3 --include-domains docs.python.org peps.python.org --exclude-domains peps.python.org

# 时间范围可选 day / week / month / year。
uv run python -m examples.web.tools --env-file .env.local search "Python release announcement" --max-results 3 --time-range month

# 不传 query：批量获取服务提取的完整正文。
uv run python -m examples.web.tools --env-file .env.local fetch https://docs.python.org/3/library/asyncio-task.html https://peps.python.org/pep-0654/

# 传 query：获取与问题相关的摘录。
uv run python -m examples.web.tools --env-file .env.local fetch https://docs.python.org/3/library/asyncio-task.html --query "How does TaskGroup cancel sibling tasks and raise ExceptionGroup?"
```

Search 返回标题、URL 和片段，零结果仍算成功。Fetch 的 `full_content` / `excerpts` 表示
请求模式；服务提取正文不等于原始 HTML，摘录不承诺固定字符上限。Tavily 两个接口固定使用 `basic`。
API 参数详见 [Web 工具说明](../../src/iris/tools/README.md#web-搜索与网页读取)。

输出超过默认 50,000 字符时，executor 保存完整内容并返回预览与文件路径。默认工作区是
`examples/web/.iris/workspace`。将输出中的真实文件路径替换下方 `ARTIFACT_PATH`：

```powershell
uv run python -m examples.web.tools read "ARTIFACT_PATH"
# 如果 has_more=true，使用输出的 next_offset / next_column 替换下面两个数值。
uv run python -m examples.web.tools read "ARTIFACT_PATH" --offset 100 --column 0
```

`read` 不调用外网，也不需要 API key。若 fetch 时指定了 `--workspace PATH`，read 时也使用
相同选项；全局选项放在 `search` / `fetch` / `read` 子命令之前。

批量失败可以这样观察：

```powershell
uv run python -m examples.web.tools --env-file .env.local fetch https://docs.python.org/3/library/asyncio.html https://docs.python.org/3/iris-web-example-missing.html
```

部分 URL 失败会同时显示正文与 `Failed URLs`，退出码为 0；全部失败、HTTP 错误等产生
`EXECUTION_ERROR`，退出码为 1。输入错误也返回非零退出码。判断批量结果时应同时查看成功数和失败数。

## 真实模型调用

[agent.yaml](agent.yaml) 启用 `web.search`、`web.fetch` 和用于续读 artifact 的 `file.read`。
[agent.py](agent.py) 用 `AgentRunner` 执行一次完整 run，最多 8 个模型步骤，并在结束后关闭 runner。

```powershell
uv run python -m examples.web.agent --env-file .env.local
```

[deepseek.yaml](deepseek.yaml) 设置 `tools.web_provider: deepseek`，将模型和 Web 工具都接到
DeepSeek。可用同一个执行入口或交互终端：

```powershell
uv run python -m examples.web.agent --env-file .env.local --config examples/web/deepseek.yaml --prompt "搜索 Python 官方文档并读取一个结果页面，解释 TaskGroup 异常处理，附来源。"
uv run iris chat examples/web/deepseek.yaml --env-file .env.local
```

两个 YAML 都显式声明相同的工具名；模型可见的参数由 Web provider 决定。

默认 Tavily 示例的问题要求搜索 Python 官方文档，读取页面摘录，解释 TaskGroup 异常处理并引用来源。
输出 `RunResult` JSON，`assistant_message` 包含回答；成功时 `run.stop_reason` 为 `completed`，
命令退出码为 0，其余运行状态退出码为 1。Tavily 示例的历史保存在
`examples/web/.iris/web.db`，DeepSeek 示例保存在 `examples/web/.iris/deepseek-web.db`。

```powershell
# 指定问题和会话；复用同一 session-id 可以继续之前的对话。
uv run python -m examples.web.agent --env-file .env.local --session-id web-demo --prompt "搜索并读取 Python 官方文档，解释 asyncio.to_thread 的用途，附来源。"

# 也可以使用同一 YAML 启动交互式终端。
uv run iris chat examples/web/agent.yaml --env-file .env.local --session-id web-chat
```

可编辑 YAML 中的模型配置，或通过 `--config` 使用另一份配置。SQLite 和 workspace 路径相对
YAML 文件所在目录解析。`.iris` 中的本地结果已被仓库忽略。

## 验证与复现

仓库仅保留使用模拟 HTTP transport 的离线测试，覆盖命令行批量失败、退出码、文件续读和
Web 工具行为，无需真实 API key：

```powershell
uv run pytest -q tests/examples/test_web_examples.py tests/tools/test_web_tools.py tests/agents/test_tools_config.py
```

需要手动验证真实服务时，可运行上面的示例命令。历史执行记录见
[真实 API 验证报告](LIVE_VALIDATION.md)。
