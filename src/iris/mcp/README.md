[English](README.en.md)

# iris.mcp

提供 MCP 外部文件导入、环境解析与官方 SDK 单服务连接。Agent YAML 与运行时装配将在后续阶段接入。

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
`AgentMCPConfig` 提供文件引用模型，当前尚未挂到 `AgentConfig`。

`resolve_server_config()` 处理 `${VAR}`、`${VAR:-default}`、`${env:VAR}`，只展开一次。
STDIO 环境优先级为 env_vars → envFile → env；不修改宿主环境。相对 cwd/envFile
基于 MCP 文件目录，未配置 cwd 则使用 workspace_root。HTTP header 在展开后按小写名称合并，
同名异值报错。HTTP、Streamable HTTP 别名及显式 SSE 都归一为内部 transport；URL 不推断 SSE。

## 单服务连接

`connection.MCPConnection(resolved)` 纯同步构造，依次 await `open()`、`list_tools()`，
再用原始名称调用 `call_tool(name, arguments)`；最后 await `aclose()`。`protocol_version`
返回实际协商结果。SDK 上下文由单个长期 task 开闭；调用直接 await session，每服务串行。
调用 timeout 包含锁等待和最多八轮 state-only continuation，取消沿调用栈传入 SDK。

依赖锁定为 MCP SDK 2.2.0、httpx2 2.12.0。STDIO、Streamable HTTP 与 SSE 使用 SDK transport，
协议由 auto 协商。真实测试已覆盖现代 STDIO（2026-07-28）与旧版 HTTP 握手（2025-11-25）；
其他组合在整体集成阶段验收。fixture 服务使用 SDK 2.2.0，旧版场景明确拒绝 discover。

SDK MCPError 转为 `IrisMCPCallError`，不推断远端是否执行；已知输出/MRTR 错误使用
`IrisMCPToolError.code`。成功结果只由 SDK 校验 output schema；2.2.0 将 null 视为缺失，
声明 output schema 时返回 `MCP_RESULT_INVALID`。不自动重放或重连。

## 实现与维护

`catalog.build_catalog(server, sdk_tools)` 按原始 wire 名过滤，编译 JSON Schema 2020-12
输入 validator；本地 `$defs`/引用可用，不能解析的外部引用和其他 dialect 留诊断后排除。
公开名使用 `mcp__...`，必要时按原始身份附稳定摘要。`tools.MCPTool(descriptor, connection)`
可直接注册到既有 ToolRegistry，参数、权限、执行与取消走普通工具链。

默认只允许本地 trust_annotations 与 readOnlyHint 同时为 true 的 MCP 工具；其他工具仍需确认。
SDK 调用不明按该只读策略回灌错误或进入 OUTCOME_UNKNOWN；Iris 主动中断未结算 claim 时
包括只读在内都沿现有 unknown 路径处理。MCP 工具首版不进入并行窗口。

富内容、structuredContent、SDK 保留的 metadata 或超长投影保存为完整 `.mcp.json`，模型只见
有界文本与路径。after middleware 扩容保留已有 JSON；仅扩容后的纯文本沿用 `.txt`。
文件按当前 context.session_id 分目录；错误的路径写入模型实际读取的 error.message。

- `config.py`：来源字段归一和环境求值的唯一入口。
- `models.py`：外部声明模型，以及内部 config/resolved/diagnostic 数据。
- `connection.py`：SDK transport owner、完整分页、调用与关闭。
- `catalog.py` / `tools.py`：descriptor 与现有 BaseTool adapter。
- `../agents/config/mcp.py`：Iris 引用与策略模型。
- `tests/mcp/test_config.py`、`test_environment.py`：复制配置、禁用、冲突与环境优先级。
- `tests/mcp/test_connection.py`、`test_sdk_contract.py`：Iris 调度与真实 SDK 契约。

在仓库根目录设置 `UV_CACHE_DIR` 后使用 `uv run pytest`、`uv run ruff check` 和
`uv run mypy`，测试范围为 `tests/mcp`。客户端 inputs、命令替换、插件变量、remote executor、
动态 header helper 与 OAuth UI 不在首版范围；启用配置包含这些字段或表达式时明确报错。
