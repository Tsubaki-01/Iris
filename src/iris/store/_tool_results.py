"""两个 store 共用的工具结果提交分类。

Example:
    claimless = is_preflight_result(result)
"""

# region imports
from ..tools.base import ToolResult

# endregion


def is_preflight_result(result: ToolResult) -> bool:
    """判断结果是否来自 executor 在 effect claim 前的明确短路。"""
    return (
        result.is_error
        and result.error is not None
        and (
            result.error.code
            in {
                "NOT_FOUND",
                "PERMISSION_ERROR",
                "TOOL_NOT_ALLOWED",
                "VALIDATION_ERROR",
                "CIRCUIT_OPEN",
            }
            or result.tool_name == "subagent"
            and result.error.code
            in {
                "SUBAGENT_CONFIG_ERROR",
                "SUBAGENT_WORKSPACE_DISJOINT",
                "SUBAGENT_PREPARE_ERROR",
            }
        )
    )
