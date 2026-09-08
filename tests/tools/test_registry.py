from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from iris.exceptions import IrisToolValidationError
from iris.tools import (
    CallableExecutionMode,
    CallableTool,
    ToolExecutionContext,
    ToolRegistry,
    tool,
)


def test_tool_registers_function_in_explicit_registry() -> None:
    """传入 registry 时，decorator 立即注册并保留原函数引用。"""
    registry = ToolRegistry()

    @tool(registry=registry, description="生成问候语")
    def greet(name: str) -> str:
        return f"你好，{name}"

    registered = registry.get("greet")

    assert isinstance(registered, CallableTool)
    assert registered.func is greet
    assert registered.definition.description == "生成问候语"
    assert greet("Iris") == "你好，Iris"


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


def test_registration_uses_existing_name_and_alias_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """注册新工具不重读已有工具定义，名称和别名冲突仍被拒绝。"""

    def first() -> str:
        return "first"

    def second() -> str:
        return "second"

    registry = ToolRegistry()
    original = CallableTool(first)
    original.definition.aliases = ("first_alias",)
    registry.register(original)
    reads: list[str] = []
    original_getattribute = CallableTool.__getattribute__

    def getattribute(tool: CallableTool, name: str) -> object:
        if tool is original and name == "definition":
            reads.append(name)
        return original_getattribute(tool, name)

    monkeypatch.setattr(CallableTool, "__getattribute__", getattribute)
    registered = registry.register_function(second)
    assert registry.get("second") is registered
    assert registry.get("first_alias") is original
    for name in ("first", "first_alias"):
        with pytest.raises(IrisToolValidationError, match="重复"):
            registry.register_function(second, name=name)
    assert reads == []


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
