from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris.message import ToolUseBlock
from iris.tools import ToolExecutionContext, ToolExecutor, ToolRegistry, ToolResult


@pytest.mark.asyncio
async def test_executor_runs_middleware_before_and_after_call(tmp_path: Path) -> None:
    class AuditMiddleware:
        async def before_call(
            self,
            tool: Any,
            params: dict[str, Any],
            context: ToolExecutionContext,
        ) -> None:
            context.metadata["before"] = f"{tool.name}:{params['name']}"

        async def after_call(
            self,
            tool: Any,
            result: ToolResult,
            context: ToolExecutionContext,
        ) -> ToolResult:
            return result.model_copy(
                update={
                    "metadata": {
                        **result.metadata,
                        "extra": {"audit": context.metadata["before"]},
                    }
                }
            )

    def greet(name: str) -> str:
        return f"你好，{name}"

    registry = ToolRegistry()
    registry.register_function(greet, description="生成问候语")

    result = await ToolExecutor(registry, middleware=[AuditMiddleware()]).execute_one(
        ToolUseBlock(id="call_1", name="greet", input={"name": "Iris"}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.metadata["extra"] == {"audit": "greet:Iris"}


@pytest.mark.asyncio
async def test_executor_maps_on_error_middleware_failure(tmp_path: Path) -> None:
    class BrokenErrorMiddleware:
        async def on_error(
            self,
            tool: Any,
            error: Exception,
            context: ToolExecutionContext,
        ) -> ToolResult | None:
            raise RuntimeError("middleware failed")

    def fail() -> str:
        raise RuntimeError("tool failed")

    registry = ToolRegistry()
    registry.register_function(fail, description="失败工具")

    result = await ToolExecutor(registry, middleware=[BrokenErrorMiddleware()]).execute_one(
        ToolUseBlock(id="call_1", name="fail", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.error is not None
    assert result.error.code == "MIDDLEWARE_ERROR"
    assert result.error.details["middleware_error"] == "middleware failed"
    assert "tool failed" in str(result.error.details["original_error"])


@pytest.mark.asyncio
async def test_executor_accepts_legacy_after_execute_only_middleware(tmp_path: Path) -> None:
    class LegacyMiddleware:
        async def after_execute(
            self,
            result: ToolResult,
            context: ToolExecutionContext,
        ) -> ToolResult:
            return result.model_copy(update={"metadata": {"extra": {"legacy": True}}})

    def greet() -> str:
        return "你好"

    registry = ToolRegistry()
    registry.register_function(greet, description="生成问候语")

    result = await ToolExecutor(registry, middleware=[LegacyMiddleware()]).execute_one(
        ToolUseBlock(id="call_1", name="greet", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.metadata["extra"] == {"legacy": True}
