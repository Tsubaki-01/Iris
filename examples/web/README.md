# Web 搜索与网页读取

这些示例通过 Iris 的 `web_search` / `web_fetch` 调用真实 Tavily API，支持独立工具调用和
真实聊天模型驱动的搜索、读取、引用回答。所有命令从仓库根目录执行。

## 配置

在 `.env.local` 中配置凭据，运行时用 `--env-file .env.local` 显式加载：

```dotenv
IRIS_PROVIDER_API_KEYS__TAVILY=tvly-你的密钥
IRIS_PROVIDER_API_KEYS__DEEPSEEK=你的密钥
```

独立工具调用只需要 Tavily；Agent 示例还需要 DeepSeek。也可以直接设置同名环境变量并省略
`--env-file`。配置通过现有 `iris.config.init_config()` 加载，不读取 `TAVILY_API_KEY`。
真实调用会消耗对应服务额度。

PowerShell 中先设置本次会话的 uv cache：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
```

## 独立工具调用

[tools.py](tools.py) 使用 `build_tool_registry()` 和 `ToolExecutor.execute_one()`，输出与模型
收到的相同 Markdown。搜索参数、错误处理和长结果 artifact 都经过真实工具执行链。
两个示例及正式 `iris chat` 命令入口均使用 UTF-8 输出，支持 Windows 管道中的网页 Unicode 字符。

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
请求模式；服务提取正文不等于原始 HTML，摘录不承诺固定字符上限。两者固定使用 `basic`。
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

默认问题要求搜索 Python 官方文档，读取搜索得到的页面摘录，解释 TaskGroup 异常处理并引用来源。
输出 `RunResult` JSON，`assistant_message` 包含回答；成功时 `run.stop_reason` 为 `completed`，
命令退出码为 0，其余运行状态退出码为 1。调用结果与历史保存在 `examples/web/.iris/web.db`。

```powershell
# 指定问题和会话；复用同一 session-id 可以继续之前的对话。
uv run python -m examples.web.agent --env-file .env.local --session-id web-demo --prompt "搜索并读取 Python 官方文档，解释 asyncio.to_thread 的用途，附来源。"

# 也可以使用同一 YAML 启动交互式终端。
uv run iris chat examples/web/agent.yaml --env-file .env.local --session-id web-chat
```

可编辑 YAML 中的模型配置，或通过 `--config` 使用另一份配置。SQLite 和 workspace 路径相对
YAML 文件所在目录解析。`.iris` 中的本地结果已被仓库忽略。

## 验证与复现

离线测试覆盖命令行批量失败、退出码和文件续读；真实测试默认跳过，显式开启后不会模拟
Tavily、聊天模型或 HTTP transport：

```powershell
uv run pytest -q -p no:cacheprovider --basetemp="$PWD\tmp\pytest-web-offline" tests/examples/test_web_examples.py tests/tools/test_web_tools.py
uv run pytest -v -s -p no:cacheprovider --basetemp="$PWD\tmp\pytest-web-live" tests/examples/test_web_live.py --run-live-web --web-env-file .env.local --junitxml=tmp/web-live.xml
```

真实测试覆盖默认搜索、数量与域名筛选、四种时间范围、空结果、批量全文、不同 query 的摘录、
部分/全部失败、artifact 逐页完整还原，以及真实模型搜索后读取并引用来源、SQLite 重读结果与历史。
工具结果和 Agent 运行证据保存在指定 `--basetemp` 的各测试目录中；相同目录下次运行会由 pytest
清理，需留存时使用不同目录。

时间范围测试确认 API 接受参数并正常返回；当前工具输出不包含可靠发布日期，不能独立验证
每条结果的严格日期。服务响应、索引和模型决策会变化，真实测试可能暴露供应商行为变化。
限流、超时、异常 JSON 等不能稳定复现的分支由现有离线测试覆盖。

本次执行记录见 [真实 API 验证报告](LIVE_VALIDATION.md)。
