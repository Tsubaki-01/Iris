"""工具执行入口。"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import ValidationError

from ..exceptions import (
    IrisToolExecutionError,
    IrisToolNotFoundError,
    IrisToolValidationError,
)
from ..message import ToolUseBlock
from .base import ToolErrorInfo, ToolExecutionContext, ToolResult
from .registry import ToolRegistry


class ToolExecutor:
    """串行执行 `ToolUseBlock` 的工具执行器。"""

    def __init__(self, registry: ToolRegistry) -> None:
        """初始化执行器。"""
        self.registry = registry

    async def execute_one(
        self,
        tool_use: ToolUseBlock,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行单个工具调用并将错误归一化为 `ToolResult`。"""
        execution_context = context.model_copy(
            update={"call_id": tool_use.id, "tool_name": tool_use.name}
        )
        try:
            tool = self.registry.get(tool_use.name)
            params = tool.validate_input(tool_use.input)
            result = await tool.arun(params, execution_context)
            return result.model_copy(
                update={
                    "tool_use_id": result.tool_use_id or tool_use.id,
                    "tool_name": result.tool_name or tool_use.name,
                }
            )
        except IrisToolNotFoundError:
            return self._error_result(tool_use, "NOT_FOUND", f"工具不存在: {tool_use.name}")
        except (IrisToolValidationError, ValidationError) as exc:
            return self._error_result(tool_use, "VALIDATION_ERROR", str(exc))
        except IrisToolExecutionError as exc:
            return self._error_result(tool_use, "EXECUTION_ERROR", exc.message)
        except Exception as exc:
            return self._error_result(tool_use, "EXECUTION_ERROR", str(exc))

    async def execute_many(
        self,
        tool_uses: Sequence[ToolUseBlock],
        context: ToolExecutionContext,
    ) -> list[ToolResult]:
        """按输入顺序串行执行多个工具调用。"""
        results: list[ToolResult] = []
        for tool_use in tool_uses:
            results.append(await self.execute_one(tool_use, context))
        return results

    def _error_result(self, tool_use: ToolUseBlock, code: str, message: str) -> ToolResult:
        """构造错误工具结果。"""
        return ToolResult(
            tool_use_id=tool_use.id,
            tool_name=tool_use.name,
            is_error=True,
            error=ToolErrorInfo(code=code, message=message),
        )
