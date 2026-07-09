"""DeepSeek live 验证脚本常量。"""

from __future__ import annotations

API_KEY_ENV_VAR = "IRIS_PROVIDER_API_KEYS__DEEPSEEK / IRIS_API_KEY"
LOCAL_ENV_FILES = (".env.local", ".env")
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_PROVIDER_ROUTE = f"deepseek/{DEFAULT_MODEL}"
PROVIDER_OK_TEXT = "IRIS_PROVIDER_OK"
RUNTIME_OK_PREFIX = "IRIS_RUNTIME_TOOL_OK:"
SCENARIO_NAMES = (
    "provider_smoke_live",
    "runtime_read_loop_live",
    "run_turn_live",
    "context_yaml_live",
    "memory_results_live",
    "memory_query_live",
    "sqlite_session_live",
    "builtin_file_tools_live",
    "file_not_read_recovery_live",
    "python_tool_live",
    "permission_path_escape_live",
    "tool_errors_live",
)
