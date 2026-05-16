from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from iris.message import ToolUseBlock
from iris.tools import ToolExecutionContext, ToolExecutor, ToolRegistry


class ExplodingPermissionPolicy:
    def check(self, tool: str, params: dict, context: ToolExecutionContext) -> None:
        raise RuntimeError("policy failed")


@pytest.mark.asyncio
async def test_executor_runs_registered_function_and_returns_text_result(tmp_path: Path) -> None:
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
