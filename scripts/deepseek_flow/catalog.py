"""DeepSeek live 验证场景目录。"""

from __future__ import annotations

SCENARIO_CATALOG: dict[str, dict[str, object]] = {
    "provider_smoke_live": {
        "module": "providers",
        "runtime_api": "direct_provider_call",
        "uses_deepseek": True,
        "description": "验证 provider factory、LiteLLM route 和响应解析。",
    },
    "runtime_read_loop_live": {
        "module": "runtime",
        "runtime_api": "run_loop",
        "uses_deepseek": True,
        "description": "验证 file.read 工具调用、tool result 回灌和最终回答。",
    },
    "run_turn_live": {
        "module": "runtime",
        "runtime_api": "run_turn",
        "uses_deepseek": True,
        "description": "验证 run_turn 一次 provider 调用和一次工具桥接语义。",
    },
    "context_yaml_live": {
        "module": "context",
        "runtime_api": "run_loop",
        "uses_deepseek": True,
        "description": "验证结构化 context.yaml 消息顺序和请求内容。",
    },
    "memory_results_live": {
        "module": "memory",
        "runtime_api": "run_loop",
        "uses_deepseek": True,
        "description": "验证显式 memory_results 注入和模型使用。",
    },
    "memory_query_live": {
        "module": "memory",
        "runtime_api": "run_loop",
        "uses_deepseek": True,
        "description": "验证 MemoryService(SQLiteMemoryStore) 写入和 memory_query 召回。",
    },
    "sqlite_session_live": {
        "module": "session",
        "runtime_api": "run_loop",
        "uses_deepseek": True,
        "description": "验证 SQLite session 的 messages、latest_run 和 tool_events。",
    },
    "builtin_file_tools_live": {
        "module": "tools.builtin.file",
        "runtime_api": "run_loop",
        "uses_deepseek": True,
        "description": "验证 list/read/grep/write/edit 内置文件工具。",
    },
    "file_not_read_recovery_live": {
        "module": "tools.builtin.file",
        "runtime_api": "run_loop",
        "uses_deepseek": True,
        "description": "验证模型收到 FILE_NOT_READ 后会先读文件再重试写入。",
    },
    "python_tool_live": {
        "module": "tools.python",
        "runtime_api": "run_loop",
        "uses_deepseek": True,
        "description": "验证 YAML 注册 Python 自定义工具和工具结果回灌。",
    },
    "permission_path_escape_live": {
        "module": "tools.permissions",
        "runtime_api": "run_loop",
        "uses_deepseek": True,
        "description": "验证 agent.yaml permissions.workspace 会拒绝父目录路径逃逸。",
    },
    "tool_errors_live": {
        "module": "tools.errors",
        "runtime_api": "run_loop",
        "uses_deepseek": True,
        "description": "验证权限拒绝、未读文件、max steps 和 tool disabled 错误路径。",
    },
}
