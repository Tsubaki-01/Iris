[English](README.en.md)

# iris.mcp

提供 MCP 外部文件导入与环境解析。当前阶段包含独立配置入口；Agent YAML 与运行时连接将在后续阶段接入。

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

## 实现与维护

- `config.py`：来源字段归一和环境求值的唯一入口。
- `models.py`：外部声明模型，以及内部 config/resolved/diagnostic 数据。
- `../agents/config/mcp.py`：Iris 引用与策略模型。
- `tests/mcp/test_config.py`、`test_environment.py`：复制配置、禁用、冲突与环境优先级。

在仓库根目录设置 `UV_CACHE_DIR` 后使用 `uv run pytest`、`uv run ruff check` 和
`uv run mypy`，测试范围为 `tests/mcp`。客户端 inputs、命令替换、插件变量、remote executor、
动态 header helper 与 OAuth UI 不在首版范围；启用配置包含这些字段或表达式时明确报错。
