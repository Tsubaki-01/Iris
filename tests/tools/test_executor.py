from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from iris.exceptions import IrisCancellationRequestedError
from iris.message import TextBlock, ToolUseBlock
from iris.tools import (
    BaseTool,
    PermissionDecision,
    PermissionEffect,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)


class _SharedCancellationSignal:
    """验证工具 context copy identity 的最小 signal。"""

    requested = False

    def raise_if_requested(self) -> None:
        return None


def test_tool_execution_context_excludes_and_shares_cancellation_signal(
    tmp_path: Path,
) -> None:
    signal = _SharedCancellationSignal()
    context = ToolExecutionContext(workspace_root=tmp_path, cancellation=signal)

    copied = context.model_copy(deep=True)

    assert "cancellation" not in context.model_dump(mode="json")
    assert copied.cancellation is signal


@pytest.mark.asyncio
async def test_callable_tool_does_not_normalize_cooperative_cancellation(
    tmp_path: Path,
) -> None:
    def cancelled() -> str:
        raise IrisCancellationRequestedError("activation 已取消")

    registry = ToolRegistry()
    registry.register_function(cancelled, description="触发取消")
    executor = ToolExecutor(registry)

    with pytest.raises(IrisCancellationRequestedError):
        await executor.execute_one(
            ToolUseBlock(id="cancel-1", name="cancelled", input={}),
            ToolExecutionContext(workspace_root=tmp_path),
        )


class ExplodingPermissionPolicy:
    def check(self, tool: str, params: dict, context: ToolExecutionContext) -> None:
        raise RuntimeError("policy failed")


class ContextCaptureMiddleware:
    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    async def before_call(
        self,
        tool: object,
        params: dict[str, str],
        context: ToolExecutionContext,
    ) -> None:
        del tool
        await asyncio.sleep(0)
        self.seen.append((params["value"], context.call_id))


class CountingPermissionPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def check(
        self,
        tool: BaseTool,
        params: dict[str, Any],
        context: ToolExecutionContext,
    ) -> PermissionDecision:
        del tool, params, context
        self.calls += 1
        return PermissionDecision(effect=PermissionEffect.ALLOW)


class CountingValidationTool(BaseTool):
    definition = ToolDefinition(
        name="counting_validation",
        description="记录每个输入的 validation 次数",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    def __init__(self) -> None:
        self.validation_calls: dict[str, int] = {}

    def validate_input(self, params: dict[str, Any]) -> dict[str, Any]:
        value = str(params["value"])
        count = self.validation_calls.get(value, 0) + 1
        self.validation_calls[value] = count
        return {"value": value, "validation_count": count}

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        assert isinstance(params, dict)
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=context.tool_name,
            content=[TextBlock(text=f"{params['value']}:{params['validation_count']}")],
        )


class ExplodingClassifierTool(CountingValidationTool):
    definition = CountingValidationTool.definition.model_copy(
        update={"name": "exploding_classifier"}
    )

    def __init__(self, executed: list[str]) -> None:
        super().__init__()
        self.executed = executed

    def is_read_only(self, params: dict[str, Any]) -> bool:
        del params
        raise RuntimeError("classifier failed")

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        del params
        self.executed.append(context.call_id)
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=context.tool_name,
            content=[TextBlock(text="ok")],
        )


@pytest.mark.asyncio
async def test_executor_runs_registered_function_and_returns_text_result(
    tmp_path: Path,
) -> None:
    def greet(name: str) -> str:
        return f"你好，{name}"

    registry = ToolRegistry()
    registry.register_function(greet, description="生成问候语")
    executor = ToolExecutor(registry)

    result = await executor.execute_one(
        ToolUseBlock(id="call_1", name="greet", input={"name": "Iris"}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.tool_use_id == "call_1"
    assert result.tool_name == "greet"
    assert result.is_error is False
    assert result.model_content == "你好，Iris"


@pytest.mark.asyncio
async def test_executor_maps_unknown_tool_to_error_result(tmp_path: Path) -> None:
    executor = ToolExecutor(ToolRegistry())

    result = await executor.execute_one(
        ToolUseBlock(id="call_1", name="missing", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "NOT_FOUND"
    assert result.model_content == "Error[NOT_FOUND]: 工具不存在: missing"


@pytest.mark.asyncio
async def test_executor_maps_validation_error_to_error_result(tmp_path: Path) -> None:
    def greet(name: str) -> str:
        return f"你好，{name}"

    registry = ToolRegistry()
    registry.register_function(greet, description="生成问候语")

    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="call_1", name="greet", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_executor_maps_callable_exception_to_error_result(tmp_path: Path) -> None:
    def fail() -> str:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register_function(fail, description="失败工具")

    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="call_1", name="fail", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "EXECUTION_ERROR"


@pytest.mark.asyncio
async def test_executor_does_not_parse_callable_exception_text_as_structured_code(
    tmp_path: Path,
) -> None:
    def fail() -> str:
        raise RuntimeError("FILE_NOT_READ: not a file tool error")

    registry = ToolRegistry()
    registry.register_function(fail, description="失败工具")

    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="call_1", name="fail", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "EXECUTION_ERROR"


@pytest.mark.asyncio
async def test_executor_runs_many_serially_in_input_order(tmp_path: Path) -> None:
    def echo(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    executor = ToolExecutor(registry)

    results = await executor.execute_many(
        [
            ToolUseBlock(id="call_1", name="echo", input={"value": "a"}),
            ToolUseBlock(id="call_2", name="echo", input={"value": "b"}),
        ],
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert [result.model_content for result in results] == ["a", "b"]


@pytest.mark.asyncio
async def test_execute_many_prepares_each_call_once(tmp_path: Path) -> None:
    tool = CountingValidationTool()
    registry = ToolRegistry()
    registry.register(tool)
    policy = CountingPermissionPolicy()
    executor = ToolExecutor(registry, permission_policy=policy)

    results = await executor.execute_many(
        [
            ToolUseBlock(id="call-1", name=tool.name, input={"value": "a"}),
            ToolUseBlock(id="call-2", name=tool.name, input={"value": "b"}),
        ],
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert tool.validation_calls == {"a": 1, "b": 1}
    assert policy.calls == 2
    assert [result.model_content for result in results] == ["a:1", "b:1"]


@pytest.mark.asyncio
async def test_execute_many_classifier_exception_falls_back_to_serial(
    tmp_path: Path,
) -> None:
    executed: list[str] = []
    tool = ExplodingClassifierTool(executed)
    registry = ToolRegistry()
    registry.register(tool)

    results = await ToolExecutor(registry).execute_many(
        [ToolUseBlock(id="call-1", name=tool.name, input={"value": "x"})],
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert results[0].is_error is False
    assert executed == ["call-1"]


@pytest.mark.asyncio
async def test_executor_uses_isolated_context_for_concurrent_read_batch(
    tmp_path: Path,
) -> None:
    def echo(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(echo, description="回显")
    middleware = ContextCaptureMiddleware()
    executor = ToolExecutor(registry, middleware=[middleware])

    await executor.execute_many(
        [
            ToolUseBlock(id="call_1", name="echo", input={"value": "a"}),
            ToolUseBlock(id="call_2", name="echo", input={"value": "b"}),
        ],
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert set(middleware.seen) == {("a", "call_1"), ("b", "call_2")}


@pytest.mark.asyncio
async def test_executor_injects_preset_kwargs_before_input_model_validation(
    tmp_path: Path,
) -> None:
    class SecretInput(BaseModel):
        query: str
        secret: str

    def search(query: str, secret: str) -> str:
        return f"{query}:{secret}"

    registry = ToolRegistry()
    registry.register_function(
        search,
        description="搜索",
        input_model=SecretInput,
        preset_kwargs={"secret": "token"},
    )

    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="call_1", name="search", input={"query": "iris"}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is False
    assert result.model_content == "iris:token"


@pytest.mark.asyncio
async def test_executor_rejects_caller_override_for_preset_kwargs(
    tmp_path: Path,
) -> None:
    class SecretInput(BaseModel):
        query: str
        secret: str

    def search(query: str, secret: str) -> str:
        return f"{query}:{secret}"

    registry = ToolRegistry()
    registry.register_function(
        search,
        description="搜索",
        input_model=SecretInput,
        preset_kwargs={"secret": "token"},
    )

    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(
            id="call_1",
            name="search",
            input={"query": "iris", "secret": "override"},
        ),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_executor_maps_permission_policy_exception_to_permission_error(
    tmp_path: Path,
) -> None:
    def greet(name: str) -> str:
        return f"你好，{name}"

    registry = ToolRegistry()
    registry.register_function(greet, description="生成问候语")

    result = await ToolExecutor(
        registry,
        permission_policy=ExplodingPermissionPolicy(),
    ).execute_one(
        ToolUseBlock(id="call_1", name="greet", input={"name": "Iris"}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "PERMISSION_ERROR"
