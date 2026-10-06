# 连接 MCP 与外部检索

已有 MCP 服务时，Iris 可以把其工具目录接入当前 Agent；只想搜索网页和读取正文时，可以直接使用内置 Web 工具。两条路径最终都返回普通工具结果，模型负责结合结果回答，宿主仍可查看调用事实和产物。

## 跑通本地 MCP 服务

仓库提供一个不需要服务端凭据的 [STDIO MCP 服务](../../examples/mcp/server.py)，包含回显和当前时间两个工具。先完成[快速开始](../getting-started/quickstart.md)并配置模型凭据，然后在仓库根目录运行：

```shell
uv run iris chat examples/mcp/agent.yaml
```

输入“调用工具，把‘你好 MCP’原样返回”，或“调用工具查询北京时间”。预期分别出现 `mcp__local__echo` 或 `mcp__local__get_current_time`；时间结果应包含 `timezone`、`iso_time`、`unix_timestamp`。时间读取自运行服务的机器，不能把示例输出当成固定值。

这条路径使用两个文件。[agent.yaml](../../examples/mcp/agent.yaml) 声明：

```yaml
mcp:
  path: mcp.json
  overrides:
    local:
      trust_annotations: true
```

[mcp.json](../../examples/mcp/mcp.json) 声明服务的启动方式：

```json
{
  "mcpServers": {
    "local": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "."
    }
  }
}
```

`mcp.path` 相对 Agent YAML 解析；这里的 `cwd` 相对 MCP 配置文件解析，因此服务从 `examples/mcp` 目录启动。省略 STDIO `cwd` 时使用 Agent workspace。`args` 由启动进程接收；其中的普通相对文件路径最终受进程 cwd 影响。

示例服务明确声明只读工具，配置选择信任其 annotations。其他服务默认不信任这一声明，其调用可能请求人工确认；这是接入时实际会遇到的差别，不要把示例的配置机械套用到所有服务。

## 接入自己的 MCP

`mcp.path` 接受 JSON、JSONC 和 TOML。STDIO 使用 `command`/`args`；Streamable HTTP 和 SSE 使用对应 `type` 与 `url`。凭据可由环境变量表达式、STDIO 的 `env_vars`/`envFile`，或 HTTP header 映射提供。支持字段、超时和环境表达式见 [MCP 参考](../reference/tools.md#mcp)。

Python 宿主使用正常 `AgentRunner` 入口：构造 runner 后可显式 `await runner.aprepare()` 提前发现服务问题；`start` 也会确保资源已准备。完成使用后 `await runner.aclose()`，释放自己启动的连接与服务。准备失败的 required 服务会阻止开始运行；optional 服务的诊断可从 MCP 目录快照读取。服务目录在准备阶段形成快照，变更服务配置或目录后重新构造、准备 runner。

模型工具名由 server 与原始工具名生成。常见结果是 `mcp__local__echo`；实际名称以准备后的目录为准，不能仅靠原服务名称猜测。启用延迟披露时先经过 `tool_search`，具体见[工具发现参考](../reference/tools.md#工具发现)。

MCP 结果会保留完整原生结果 artifact。普通文本进入模型正文，支持的图片进入图片块；其他资源内容仍可留在 artifact 中供宿主检查。获得 MCP artifact 路径不代表每一种二进制内容都已经送入模型。

无需真实模型的本地互通验证位于 [MCP 示例测试](../../tests/examples/test_mcp_examples.py)，它用固定模型响应启动真实 STDIO 服务，检查实际调用与保存的原生结果。它验证协议和工具链路，不评估模型能否自主选对工具。

## 直接搜索网页

内置 `web.search` 与 `web.fetch` 使用 Tavily。配置 `IRIS_PROVIDER_API_KEYS__TAVILY` 后，在仓库根目录可以先运行不依赖主模型的工具示例：

```shell
uv run python -m examples.web.tools search "Python asyncio TaskGroup" --include-domains docs.python.org --max-results 3
```

成功结果包含标题、URL 和来源片段。复制其中的完整 URL，再调用：

```shell
uv run python -m examples.web.tools fetch "https://docs.python.org/3/library/asyncio-task.html" --query "TaskGroup cancellation"
```

这里的 URL 是一个具体调用例子；搜索结果可能随时间变化。`query` 对一批 URL 生效，用于获取相关摘录；省略它时返回服务提取的完整正文。它不是浏览器页面渲染，也不执行交互操作。

`web_fetch` 可一次提交 1–20 个 URL。部分页面失败时，成功正文和各 URL 的失败原因一起返回；全部提取失败时为工具错误。搜索或正文过长时，结果提供 artifact 正文路径，继续运行 `examples.web.tools read <返回路径>`，按返回的 `next_offset`、`next_column` 分页。

## 交给 Agent 完成搜索到回答

[Web Agent 示例](../../examples/web/agent.yaml) 同时声明 `web.search`、`web.fetch`、`file.read`，并在 prompt 中要求先搜索、读取来源、再附链接回答。配置模型与 Tavily 凭据后运行：

```shell
uv run iris chat examples/web/agent.yaml
```

成功标志是可看到真实搜索和正文读取调用，并得到有来源链接的回答；答案措辞由模型决定。工具示例的 HTTP 连接成功、Agent 的工具调用成功和答案质量是三个不同的验证目标。

后续阅读：[工具结果与 artifact](tools.md)、[观察与排错](observability.md)、[MCP 与 Web 参数参考](../reference/tools.md)。源码入口：[MCP 配置解析](../../src/iris/mcp/config.py)、[MCP 工具投影](../../src/iris/mcp/tools.py)、[Web 工具](../../src/iris/tools/builtin/web.py)。
