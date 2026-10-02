[English](README.en.md)

# iris.mcp

提供 MCP 外部文件导入、官方 SDK 连接、多服务目录发布与工具适配，已接入 Agent YAML、root/child runner 与 CLI。

可运行的本地服务与 JSON/TOML 配置见 [examples/mcp](../../../examples/mcp/README.md)。
稳定包级入口为 `MCPManager` 与 `load_mcp_config`；配置模型从 `iris.agents` 导出。

## 配置入口

Python 3.12+；依赖随 Iris 安装。`config.load_mcp_config()` 接受 JSON/JSONC 的
`mcpServers`、`servers`、`mcp_servers`，以及 TOML 的 `[mcp_servers.<name>]`。
只导入一个明确文件的 MCP 区段。原始 server 名保留大小写和连字符。

```python
import os
from pathlib import Path

from iris.mcp.config import load_mcp_config, resolve_server_config

config = load_mcp_config(Path("mcp.json"), overrides={})
servers = [
    resolve_server_config(server, environ=os.environ, workspace_root=Path.cwd())
    for server in config.servers
]
```

`load_mcp_config()` 只读文件并校验声明，不连接服务；禁用项跳过其余字段。
required 默认为 true，无效声明抛 `IrisConfigError`；optional 项进入 `config.diagnostics`。
`agents.config.mcp.MCPServerOverride` 支持 required、trust_annotations 和两个 timeout 覆盖项。
`AgentConfig.mcp` 使用 `AgentMCPConfig`；`mcp.path` 相对 agent YAML 解析。
shared assembly 读取声明；root `AgentRunner.aprepare()` 在首次执行前准备并发布完整目录。
连接跨 run 复用，host 等所有原执行调用完整结束后 await `runner.aclose()`。
child 使用相同 YAML 配置和独立连接，admission 前准备，WAITING/结束后关闭；恢复按当前配置
重新发现工具。child 关闭不影响 root；权限由既有父子组合策略裁决。

`context_policy.deferred_tools: true` 将 MCP 工具定义标记为 deferred，由自动注册的
`tool_search` 发现，成功结果提交后再为下一请求选择完整 schema。默认 `false` 保持 eager。
这只减少模型请求中的 schema：所有服务仍在执行前完整连接、发现和发布，连接生命周期与
required/optional 准备失败语义不变。预算内的可见集合由 runtime 按 session/步骤选择，
不会卸载连接或更改 MCP 目录。见 [按需工具说明](../runtime/README.md#按需工具定义)。

`resolve_server_config()` 处理 `${VAR}`、`${VAR:-default}`、`${env:VAR}`，只展开一次。
STDIO 环境优先级为 env_vars → envFile → env；不修改宿主环境。相对 cwd/envFile
基于 MCP 文件目录，未配置 cwd 则使用 workspace_root。HTTP header 在展开后按小写名称合并，
同名异值报错。HTTP、Streamable HTTP 别名及显式 SSE 都归一为内部 transport；URL 不推断 SSE。

## 单服务连接

`connection.MCPConnection(resolved)` 纯同步构造，依次 await `open()`、`list_tools()`，
再用原始名称调用 `call_tool(name, arguments)`；最后 await `aclose()`。`protocol_version`
返回实际协商结果。SDK 上下文由单个长期 task 开闭；调用直接 await session，每服务串行。
调用 timeout 包含锁等待和最多八轮 state-only continuation，取消沿调用栈传入 SDK。
SSE 连接的空闲等待不计入工具调用期限；准备和调用各自受外层期限约束。

依赖锁定为 MCP SDK 2.2.0、httpx2 2.12.0。STDIO、Streamable HTTP 与 SSE 使用 SDK transport，
协议由 auto 协商。真实 `AgentRunner → executor → SDK → server` 测试覆盖以下组合：

| transport | 协议 | 本地 fixture |
| --- | --- | --- |
| STDIO / Streamable HTTP | 2026-07-28 | SDK 2.2.0 Server，现代 discover |
| STDIO / Streamable HTTP | 2025-11-25 | SDK handshake-only driver / 旧版 HTTP 初始化 |
| 显式 SSE | 2025-11-25 | SDK handshake-only driver |

fixture server 为 iris-test v1；另有现代 SSE 空闲复用回归。HTTP 验证实际静态/env header
与 client 关闭，STDIO 验证进程退出；无公网 MCP 服务或真实 LLM 费用。

SDK MCPError 转为 `IrisMCPCallError`，不推断远端是否执行；已知输出/MRTR 错误使用
`IrisMCPToolError.code`。成功结果只由 SDK 校验 output schema；2.2.0 将 null 视为缺失，
声明 output schema 时返回 `MCP_RESULT_INVALID`。不自动重放或重连。

## 实现与维护

`manager.MCPManager(config, registry=registry, workspace_root=workspace, defer_tools=False)` 的 `prepare()` 按
server id 顺序解析环境、连接和完整发现。每服务共享一个 startup 期限，全部成功候选通过
`registry.register_many()` 一次发布，原有 ToolRegistryView 随即可读取完整目录；deferred 项仍需
模型步骤选中才导出 schema。required 失败不发布
部分工具，optional 失败记录 diagnostic；合法空目录允许继续。

`snapshot` 在成功准备后固定；并发/重复 prepare 不重复连接或发现。失败后 manager 关闭，
需要新配置时构造新实例。`aclose()` 尝试关闭所有 owned connection，显式关闭错误报告 host。
快照中的有效 env/header 保留在内存中，不应整体记录或持久化。

`catalog.build_catalog(server, sdk_tools, defer_tools=False)` 按原始 wire 名过滤，编译 JSON Schema 2020-12
输入 validator；本地 `$defs`/引用可用，不能解析的外部引用和其他 dialect 留诊断后排除。
公开名使用 ASCII 字母、数字和下划线组成的 `mcp__...`，必要时按原始身份附稳定摘要。
`tools.MCPTool(descriptor, connection)`
可直接注册到既有 ToolRegistry，参数、权限、执行与取消走普通工具链。
shared assembly 将 context policy 的开关作为同一个 `defer_tools` 参数传给 manager/catalog，
不改变远端 schema 或按需重建 adapter。

默认只允许本地 trust_annotations 与 readOnlyHint 同时为 true 的 MCP 工具；其他工具仍需确认。
SDK 调用不明按该只读策略回灌错误，或抛公共 `IrisToolOutcomeUnknownError` 进入
OUTCOME_UNKNOWN；Iris 主动中断未结算 claim 时
包括只读在内都沿现有 unknown 路径处理。MCP 工具首版不进入并行窗口。
取消通过现有 executor 传入 SDK，请求清理结束后才结算；这不证明远端副作用已经终止。
CLI 退出先等待 `manager.close(cancel_run=True)`，再关闭 runner，最后收尾输出和后台 loop。

`TextContent` 与 `ImageContent` 按原顺序投影为文字/图片块，保留同一 tool call ID。图片的
base64 在 MCP 结果边界解码，再通过 `tools.images.import_tool_image` 处理和保存到当前
session 的 `.iris/image-cache/` 子目录；实际 PNG/JPEG/WebP 格式由公共图片处理器识别，
不采信远端声明的 MIME。模型读取已准备的 model 副本，original 保留原始字节；MCP 层不
选择 API 图片格式。音频、资源链接、嵌入资源与 structuredContent 保持文字预览。

富内容、structuredContent、SDK 保留的 metadata 或超长投影仍保存为完整 `.mcp.json`，模型
收到有界文字、图片块与结果文件路径。after middleware 后文本仍超限时，保留 JSON 的
`artifact.path`，另写完整 `.model.txt` 到 `artifact.text_path`；没有原生产物的大文本只写
`.txt`，两字段指向同一文件。
文字裁剪不移除图片；text_path 开头包含图片副本引用，后面保存完整模型文字。
原生 MCP JSON 与最终模型文本分开保存，不能互相替代。文件按当前 `context.session_id`
分目录，每次落盘使用新的随机标识；错误的路径写入模型实际读取的 `error.message`。
远端 `isError` 结果保留所有图片，通过唯一错误正文投影输出一次错误说明；只有图片的错误
结果使用明确的业务错误文字。
adapter 保留完整文本供 middleware 消费，最终裁剪统一由 executor 完成；预览长度取当前
ToolDefinition.preview_chars。直接调用 MCPTool.arun 得到的是尚未经过 executor 限长的结果。
启用默认 context policy 的完整 runner 会附加历史 result 引用；`context_read` 的 `text`
表示读取最终模型文本，`raw` 表示读取原生 MCP JSON，无需重新调用远端。范围与分页见
[上下文回读](../tools/README.md#当前会话上下文回读)。
SDK 返回后，图片导入、完整投影、`model_dump`、JSON 编码和落盘在同一次工具 IO worker 中
执行。取消只延后到本地作业实际完成；图片解码/保存失败成为 `IMAGE_ERROR`，JSON 落盘失败
保持 `ARTIFACT_ERROR`。这些都是已取得远端结果后的确定本地失败，不进入 outcome-unknown，
也不重新调用远端；SDK 网络等待仍可取消。

- `config.py`：来源字段归一和环境求值的唯一入口。
- `models.py`：外部声明模型，以及内部 config/resolved/diagnostic 数据。
- `connection.py`：SDK transport owner、完整分页、调用与关闭。
- `catalog.py` / `tools.py`：descriptor 与现有 BaseTool adapter。
- `manager.py`：准备/关闭协调与唯一 catalog snapshot。
- `../agents/config/mcp.py`：Iris 引用与策略模型。
- `tests/mcp/test_config.py`、`test_environment.py`：复制配置、禁用、冲突与环境优先级。
- `tests/mcp/test_connection.py`、`test_sdk_contract.py`：Iris 调度与真实 SDK 契约。
- `tests/mcp/test_interoperability.py`：五行完整互操作与公开取消后新 run 复用。

在仓库根目录设置 `UV_CACHE_DIR` 后使用 `uv run pytest`、`uv run ruff check` 和
`uv run mypy`，测试范围为 `tests/mcp`。客户端 inputs、命令替换、插件变量、remote executor、
动态 header helper 与 OAuth UI 不在首版范围；启用配置包含这些字段或表达式时明确报错。
