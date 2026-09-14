[English](README.en.md)

# 本地 MCP 示例

从 Iris 仓库根目录运行，使用 Python 3.12+ 与已同步的 uv 环境。服务使用随 Iris 安装的
官方 MCP SDK，提供只读回显和本机时间查询；服务本身无需外部 API 或凭据。

| 工具 | 输入 | 返回值 |
| --- | --- | --- |
| `echo` | `text` | 原样回显文本 |
| `get_current_time` | `timezone_name`，默认 `Asia/Shanghai`，也支持 `UTC` | `timezone`、带偏移量的 `iso_time`、整数 `unix_timestamp` |

时间工具直接读取本机系统时钟，精确到秒。北京时间使用当前的 UTC+08:00；不提供历史时间或其他时区转换。
不支持的时区由工具 schema 拒绝。时间结果同时通过 MCP 结构化结果和 Iris JSON artifact 保留。

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv sync
uv run iris chat examples/mcp/agent.yaml
```

实际聊天需要配置 DeepSeek key（或修改 agent.yaml 的 model）：
`IRIS_PROVIDER_API_KEYS__DEEPSEEK`。输入“请用 MCP 回显：你好”。无需手动启动 server.py；
Iris 首次执行前启动服务，后续对话复用连接，退出时等待正在运行的请求收尾后关闭进程。

查询时间可以输入“请通过 MCP 查询当前北京时间，再查询 UTC 时间”。模型会调用
`mcp__local__get_current_time`，读取服务返回值；时间准确性取决于本机系统时钟。

`mcp.json` 与 Codex 格式的 `mcp.toml` 指向同一服务。把 agent.yaml 的 `mcp.path` 改为
`mcp.toml` 即可切换。两者 `cwd: .` 都相对 MCP 文件目录，因而 Python 执行 examples/mcp/server.py。
`uv run` 提供当前环境的 Python；从其他 host 启动时，确保 PATH 中的 Python 已安装 Iris 依赖，
也可以将 command 改为该解释器的绝对路径。程序不存在时，required server 准备失败，不创建 run。

本地只读信任放在 agent.yaml 的 overrides 中；远端 readOnlyHint 只有与本地
trust_annotations 同时启用才允许默认自动调用。其他工具使用普通权限/HITL。

不调用真实 LLM，也可以验证两个配置实际启动、调用、关闭：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run pytest -p no:cacheprovider --basetemp="$PWD\tmp\pytest-mcp-example-check" tests/examples/test_mcp_examples.py tests/examples/test_mcp_time.py
```

重复手动运行时请选择新的 basetemp 目录。更多配置、结果 JSON artifact、恢复和限制见
[MCP 包说明](../../src/iris/mcp/README.md)，SDK 生命周期见
[harness 使用说明](../../src/iris/harness/README.md)。
