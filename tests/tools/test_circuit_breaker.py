from __future__ import annotations

from pathlib import Path

import pytest

from iris.message import ToolUseBlock
from iris.tools import CircuitBreaker, ToolExecutionContext, ToolExecutor, ToolRegistry


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_consecutive_failures(tmp_path: Path) -> None:
    calls = 0

    def unstable() -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register_function(unstable, description="不稳定工具")
    executor = ToolExecutor(
        registry,
        circuit_breaker=CircuitBreaker(failure_threshold=2, cooldown_seconds=60),
    )

    for _ in range(2):
        result = await executor.execute_one(
            ToolUseBlock(id="call_1", name="unstable", input={}),
            ToolExecutionContext(workspace_root=tmp_path),
        )
        assert result.error is not None
        assert result.error.code == "EXECUTION_ERROR"

    result = await executor.execute_one(
        ToolUseBlock(id="call_2", name="unstable", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert calls == 2
    assert result.error is not None
    assert result.error.code == "CIRCUIT_OPEN"
