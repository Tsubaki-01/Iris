from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable
from pathlib import Path

import pytest

import iris.tools as tools
from iris.exceptions import IrisToolValidationError
from iris.tools import (
    CallableExecutionMode,
    ToolExecutionContext,
    ToolRegistry,
    tool,
)


def test_function_registration_exports_schema_from_type_hints() -> None:
    def greet(name: str, excited: bool = False) -> str:
        """生成问候语。"""
        suffix = "!" if excited else "."
        return f"你好，{name}{suffix}"

    registry = ToolRegistry()
    registry.register_function(greet, description="生成问候语")

    assert registry.active_schemas() == [
        {
            "name": "greet",
            "description": "生成问候语",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "excited": {"type": "boolean", "default": False},
                },
                "required": ["name"],
            },
        }
    ]


def test_callable_execution_mode_is_public() -> None:
    """工具作者可以从公共 tools 包显式选择 callable placement。"""
    execution_mode = getattr(tools, "CallableExecutionMode", None)

    assert execution_mode is not None
    assert execution_mode.INLINE.value == "inline"
    assert execution_mode.THREAD.value == "thread"


@pytest.mark.asyncio
async def test_callable_default_stays_inline() -> None:
    """未声明 placement 的同步函数仍在事件循环线程执行。"""
    loop_thread_id = threading.get_ident()

    def current_thread_id() -> int:
        return threading.get_ident()

    registry = ToolRegistry()
    registered = registry.register_function(current_thread_id)

    result = await registered.arun(
        registered.validate_input({}),
        ToolExecutionContext(workspace_root=Path.cwd()),
    )

    assert result.model_content == str(loop_thread_id)
    assert result.stats["execution_mode"] == "inline"
    assert "execution_mode" not in registered.definition.metadata
    assert registered.is_concurrency_safe({}) is True


@pytest.mark.asyncio
async def test_callable_explicit_thread_runs_on_worker() -> None:
    """显式 thread placement 把同步函数移出事件循环线程。"""
    loop_thread_id = threading.get_ident()

    def current_thread_id() -> int:
        return threading.get_ident()

    registry = ToolRegistry()
    registered = registry.register_function(
        current_thread_id,
        execution_mode=CallableExecutionMode.THREAD,
    )

    result = await registered.arun(
        registered.validate_input({}),
        ToolExecutionContext(workspace_root=Path.cwd()),
    )

    assert result.model_content != str(loop_thread_id)
    assert result.stats["execution_mode"] == "thread"
    assert registered.definition.metadata["execution_mode"] == "thread"
    provider_schema = registry.active_schemas()[0]
    assert "execution_mode" not in provider_schema
    assert "concurrency_safe" not in provider_schema


@pytest.mark.asyncio
async def test_thread_callable_keeps_event_loop_responsive_and_can_time_out() -> None:
    """worker 未结束时 loop 仍可推进，等待超时也不伪装成 worker 已终止。"""
    started = threading.Event()
    release = threading.Event()

    def blocking_call() -> str:
        started.set()
        release.wait(timeout=2)
        return "done"

    registry = ToolRegistry()
    registered = registry.register_function(
        blocking_call,
        execution_mode=CallableExecutionMode.THREAD,
    )
    execution = asyncio.create_task(
        registered.arun(
            registered.validate_input({}),
            ToolExecutionContext(workspace_root=Path.cwd()),
        )
    )

    try:
        assert await asyncio.to_thread(started.wait, 1)
        for _ in range(3):
            await asyncio.sleep(0)
        assert not execution.done()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(execution), timeout=0.01)
        assert not execution.done()
    finally:
        release.set()
        await execution


@pytest.mark.asyncio
async def test_thread_callable_awaits_awaitable_result_on_loop() -> None:
    """同步 factory 在线程返回 awaitable 后仍沿用既有归一化语义。"""

    async def resolve_value() -> str:
        return "resolved"

    def awaitable_factory() -> Awaitable[str]:
        return resolve_value()

    registry = ToolRegistry()
    registered = registry.register_function(
        awaitable_factory,
        execution_mode=CallableExecutionMode.THREAD,
    )

    result = await registered.arun(
        registered.validate_input({}),
        ToolExecutionContext(workspace_root=Path.cwd()),
    )

    assert result.model_content == "resolved"


def test_callable_rejects_string_execution_mode() -> None:
    """placement 不接受与 enum 同值但类型不受控的字符串。"""

    def read_value() -> str:
        return "ok"

    registry = ToolRegistry()

    with pytest.raises(IrisToolValidationError, match="CallableExecutionMode"):
        registry.register_function(
            read_value,
            execution_mode="thread",  # type: ignore[arg-type]
        )

    assert registry.active_schemas() == []


def test_callable_rejects_async_function_in_thread_mode() -> None:
    """async callable 不能被错误提交到同步线程 placement。"""

    async def async_tool() -> str:
        return "ok"

    registry = ToolRegistry()

    with pytest.raises(IrisToolValidationError, match="async callable"):
        registry.register_function(
            async_tool,
            execution_mode=CallableExecutionMode.THREAD,
        )

    assert registry.active_schemas() == []


def test_callable_concurrency_declaration_overrides_default() -> None:
    """显式并发声明改变本地分类并进入恢复指纹元数据。"""

    def read_value() -> str:
        return "ok"

    registry = ToolRegistry()
    registered = registry.register_function(read_value, concurrency_safe=False)

    assert registered.is_concurrency_safe({}) is False
    assert registered.definition.metadata["concurrency_safe"] is False


@pytest.mark.asyncio
async def test_register_arguments_override_decorator_placement() -> None:
    """register_function 显式参数优先于 decorator 声明。"""
    loop_thread_id = threading.get_ident()

    @tool(
        execution_mode=CallableExecutionMode.THREAD,
        concurrency_safe=False,
    )
    def current_thread_id() -> int:
        return threading.get_ident()

    registry = ToolRegistry()
    registered = registry.register_function(
        current_thread_id,
        execution_mode=CallableExecutionMode.INLINE,
        concurrency_safe=True,
    )

    result = await registered.arun(
        registered.validate_input({}),
        ToolExecutionContext(workspace_root=Path.cwd()),
    )

    assert result.model_content == str(loop_thread_id)
    assert result.stats["execution_mode"] == "inline"
    assert "execution_mode" not in registered.definition.metadata
    assert registered.definition.metadata["concurrency_safe"] is True
    assert registered.is_concurrency_safe({}) is True
