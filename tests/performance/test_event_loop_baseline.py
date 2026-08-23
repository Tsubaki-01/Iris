"""P1 callable 执行边界的本机前后观测。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import iris.tools as tools


async def _measure_callable(
    tool: tools.CallableTool,
    *,
    samples: int,
) -> tuple[tuple[float, ...], int]:
    elapsed_samples: list[float] = []
    heartbeat_ticks = 0

    for _ in range(samples):
        stop = asyncio.Event()

        async def heartbeat(stop_signal: asyncio.Event) -> None:
            nonlocal heartbeat_ticks
            while not stop_signal.is_set():
                await asyncio.sleep(0.005)
                heartbeat_ticks += 1

        heartbeat_task = asyncio.create_task(heartbeat(stop))
        started_at = time.perf_counter()
        await tool.arun(
            {},
            tools.ToolExecutionContext(workspace_root=Path.cwd()),
        )
        elapsed_samples.append((time.perf_counter() - started_at) * 1000)
        stop.set()
        await heartbeat_task

    return tuple(elapsed_samples), heartbeat_ticks


@pytest.mark.performance_timing
@pytest.mark.asyncio
async def test_p1_callable_execution_observation(
    require_performance_timing: None,
    record_observation: Callable[..., None],
) -> None:
    """记录默认 inline 与可用时显式 thread 的响应性形状。"""

    def blocking_callable() -> str:
        time.sleep(0.08)
        return "ok"

    inline_samples, inline_ticks = await _measure_callable(
        tools.CallableTool(blocking_callable),
        samples=5,
    )
    assert inline_ticks == 0
    record_observation(
        scenario="p1_callable_inline",
        perf_ids=("PERF-01", "PERF-14"),
        fixture={"sleep_ms": 80, "heartbeat_ms": 5, "execution_mode": "inline"},
        samples_ms=inline_samples,
        counters={"heartbeat_ticks": inline_ticks, "thread_mode_supported": 0},
    )

    execution_mode_type: Any = getattr(tools, "CallableExecutionMode", None)
    if execution_mode_type is None:
        return

    thread_samples, thread_ticks = await _measure_callable(
        tools.CallableTool(
            blocking_callable,
            execution_mode=execution_mode_type.THREAD,
        ),
        samples=5,
    )
    assert thread_ticks > 0
    record_observation(
        scenario="p1_callable_thread",
        perf_ids=("PERF-01", "PERF-14"),
        fixture={"sleep_ms": 80, "heartbeat_ms": 5, "execution_mode": "thread"},
        samples_ms=thread_samples,
        counters={"heartbeat_ticks": thread_ticks, "thread_mode_supported": 1},
    )
