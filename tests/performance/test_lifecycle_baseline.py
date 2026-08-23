"""P2 lifecycle hot-read 的结构计数与同机计时观测。"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

import iris.store.sqlite as sqlite_module
from iris.harness._commit_port import StoreRuntimeCommitPort
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    CreateRun,
    RunCheckpoint,
    RunToolCallRecord,
)
from iris.message import Msg, ToolUseBlock
from iris.runtime import RuntimeCursor
from iris.runtime.commit import RuntimeModelStepCommit, RuntimeToolCall
from iris.store import InMemoryLifecycleStore, SQLiteStore

_NOW = datetime(2026, 8, 20, tzinfo=UTC)
_FINGERPRINT = "a" * 64


class _CountingToolCalls(dict[tuple[str, str], RunToolCallRecord]):
    """只统计 global values traversal，不改变 dict 读取语义。"""

    def __init__(self, source: dict[tuple[str, str], RunToolCallRecord]) -> None:
        super().__init__(source)
        self.values_calls = 0
        self.values_yielded = 0

    def values(self) -> Iterator[RunToolCallRecord]:  # type: ignore[override]
        self.values_calls += 1
        for record in super().values():
            self.values_yielded += 1
            yield record


def _create_port(
    path: Path,
) -> tuple[SQLiteStore, StoreRuntimeCommitPort, RuntimeCursor]:
    store = SQLiteStore(path)
    before = RuntimeCursor(position="before_model", step_index=0)
    created = store.create_run(
        CreateRun(
            request=AgentRunRequest(input="hello", session_id="session_1", run_id="run_1"),
            options=AgentRunOptions(),
            agent_id="agent_1",
            environment_fingerprint=_FINGERPRINT,
            start_activation_id="activation_1",
            initial_checkpoint=RunCheckpoint(
                run_id="run_1",
                sequence=1,
                activation_id="activation_1",
                engine_cursor=before.model_dump(mode="json"),
                session_revision=0,
                model_steps_reserved=0,
                model_steps_committed=0,
                environment_fingerprint=_FINGERPRINT,
            ),
            now=_NOW,
        )
    )
    port = StoreRuntimeCommitPort(
        store=store,
        run=created.run,
        activation_id="activation_1",
        clock=lambda: _NOW,
        event_sink=[],
    )
    port.reserve_model_step(before)
    return store, port, before


def _install_read_counters(
    stack: ExitStack,
    store: SQLiteStore,
) -> dict[str, int]:
    counters = {
        "connections": 0,
        "full_run_decodes": 0,
        "load_run": 0,
        "load_run_control": 0,
        "list_tool_calls": 0,
        "load_tool_call": 0,
    }
    original_connect = store._connect
    original_decode = sqlite_module._row_to_run
    original_load_run = store.load_run
    original_list_tool_calls = store.list_tool_calls

    def connect():
        counters["connections"] += 1
        return original_connect()

    def decode(row):
        counters["full_run_decodes"] += 1
        return original_decode(row)

    def load_run(run_id: str):
        counters["load_run"] += 1
        return original_load_run(run_id)

    def list_tool_calls(run_id: str):
        counters["list_tool_calls"] += 1
        return original_list_tool_calls(run_id)

    stack.enter_context(patch.object(store, "_connect", connect))
    stack.enter_context(patch.object(sqlite_module, "_row_to_run", decode))
    stack.enter_context(patch.object(store, "load_run", load_run))
    stack.enter_context(patch.object(store, "list_tool_calls", list_tool_calls))
    if hasattr(store, "load_run_control"):
        original_load_run_control = store.load_run_control

        def load_run_control(run_id: str):
            counters["load_run_control"] += 1
            return original_load_run_control(run_id)

        stack.enter_context(patch.object(store, "load_run_control", load_run_control))
    if hasattr(store, "load_tool_call"):
        original_load_tool_call = store.load_tool_call

        def load_tool_call(run_id: str, tool_call_id: str):
            counters["load_tool_call"] += 1
            return original_load_tool_call(run_id, tool_call_id)

        stack.enter_context(patch.object(store, "load_tool_call", load_tool_call))
    return counters


def _measure_text_commit(path: Path) -> tuple[float, dict[str, int]]:
    store, port, before = _create_port(path)
    assistant = Msg.assistant("done")
    commit = RuntimeModelStepCommit(
        cursor_before=before,
        message_delta=(assistant,),
        assistant_message=assistant,
        cursor_after=RuntimeCursor(
            position="outcome_ready",
            step_index=0,
            assistant_message=assistant,
        ),
    )
    with ExitStack() as stack:
        counters = _install_read_counters(stack, store)
        started_at = time.perf_counter()
        port.commit_model_step(commit)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
    return elapsed_ms, counters


def _measure_tool_claim(path: Path) -> tuple[float, dict[str, int]]:
    store, port, before = _create_port(path)
    tool_use = ToolUseBlock(id="call_1", name="echo", input={"value": "hello"})
    assistant = Msg.assistant([tool_use])
    call = RuntimeToolCall(
        run_id="run_1",
        activation_id="activation_1",
        step_index=0,
        ordinal=1,
        tool_call_id=tool_use.id,
        tool_name=tool_use.name,
        arguments=dict(tool_use.input),
        fingerprint=_FINGERPRINT,
    )
    port.commit_model_step(
        RuntimeModelStepCommit(
            cursor_before=before,
            message_delta=(assistant,),
            assistant_message=assistant,
            prepared_tool_calls=(call,),
            cursor_after=RuntimeCursor(
                position="tool_batch",
                step_index=0,
                tool_calls=(tool_use,),
                assistant_message=assistant,
            ),
        )
    )
    with ExitStack() as stack:
        counters = _install_read_counters(stack, store)
        started_at = time.perf_counter()
        port.claim_tool_call(call)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
    return elapsed_ms, counters


def _measure_in_memory_target_list(
    *,
    global_scan_reference: bool = False,
) -> tuple[float, dict[str, int]]:
    store = InMemoryLifecycleStore()
    before = RuntimeCursor(position="before_model", step_index=0)
    store.create_run(
        CreateRun(
            request=AgentRunRequest(input="hello", session_id="session_1", run_id="run_1"),
            options=AgentRunOptions(),
            agent_id="agent_1",
            environment_fingerprint=_FINGERPRINT,
            start_activation_id="activation_1",
            initial_checkpoint=RunCheckpoint(
                run_id="run_1",
                sequence=1,
                activation_id="activation_1",
                engine_cursor=before.model_dump(mode="json"),
                session_revision=0,
                model_steps_reserved=0,
                model_steps_committed=0,
                environment_fingerprint=_FINGERPRINT,
            ),
            now=_NOW,
        )
    )
    records = [
        RunToolCallRecord(
            run_id="run_1" if index == 0 else f"other-{index}",
            step_index=0,
            ordinal=1,
            tool_call_id=f"call-{index}",
            tool_name="echo",
            arguments={"index": index},
            fingerprint=_FINGERPRINT,
            phase="prepared",
            version=1,
            created_at=_NOW,
            updated_at=_NOW,
        )
        for index in range(1001)
    ]
    with store._lock:
        if hasattr(store, "_set_tool_call"):
            for record in records:
                store._set_tool_call(record)
        else:
            store._tool_calls.update(
                ((record.run_id, record.tool_call_id), record) for record in records
            )
        counted = _CountingToolCalls(store._tool_calls)
        store._tool_calls = counted

    started_at = time.perf_counter()
    if global_scan_reference:
        result = deepcopy(
            sorted(
                (record for record in counted.values() if record.run_id == "run_1"),
                key=lambda record: (record.step_index, record.ordinal),
            )
        )
    else:
        result = store.list_tool_calls("run_1")
    elapsed_ms = (time.perf_counter() - started_at) * 1000

    assert [record.tool_call_id for record in result] == ["call-0"]
    return elapsed_ms, {
        "global_values_calls": counted.values_calls,
        "global_records_yielded": counted.values_yielded,
        "target_records": len(result),
    }


def test_p2_sqlite_text_commit_avoids_full_control_decode_and_empty_tool_list(
    tmp_path: Path,
) -> None:
    _, counters = _measure_text_commit(tmp_path / "text.db")

    assert counters == {
        "connections": 2,
        "full_run_decodes": 1,
        "load_run": 0,
        "load_run_control": 1,
        "list_tool_calls": 0,
        "load_tool_call": 0,
    }


def test_p2_sqlite_tool_claim_uses_control_and_point_reads(tmp_path: Path) -> None:
    _, counters = _measure_tool_claim(tmp_path / "tool.db")

    assert counters == {
        "connections": 3,
        "full_run_decodes": 1,
        "load_run": 0,
        "load_run_control": 1,
        "list_tool_calls": 0,
        "load_tool_call": 1,
    }


def test_p2_in_memory_list_uses_only_target_run_index() -> None:
    _, counters = _measure_in_memory_target_list()

    assert counters == {
        "global_values_calls": 0,
        "global_records_yielded": 0,
        "target_records": 1,
    }


@pytest.mark.performance_timing
@pytest.mark.parametrize("scenario", ["text", "tool"])
def test_p2_lifecycle_hot_read_observation(
    tmp_path: Path,
    require_performance_timing: None,
    record_observation: Callable[..., None],
    scenario: str,
) -> None:
    samples: list[float] = []
    totals: dict[str, int] = {}
    measure = _measure_text_commit if scenario == "text" else _measure_tool_claim
    for index in range(5):
        elapsed_ms, counters = measure(tmp_path / f"{scenario}-{index}.db")
        samples.append(elapsed_ms)
        for name, value in counters.items():
            totals[name] = totals.get(name, 0) + value

    record_observation(
        scenario=f"p2_sqlite_{scenario}_hot_reads",
        perf_ids=("PERF-04", "PERF-07"),
        fixture={"samples": 5, "tool_path": scenario == "tool"},
        samples_ms=tuple(samples),
        counters=totals,
    )


@pytest.mark.performance_timing
@pytest.mark.parametrize("global_scan_reference", [True, False])
def test_p2_in_memory_list_observation(
    require_performance_timing: None,
    record_observation: Callable[..., None],
    global_scan_reference: bool,
) -> None:
    elapsed_ms, counters = _measure_in_memory_target_list(
        global_scan_reference=global_scan_reference
    )
    algorithm = "global_scan_reference" if global_scan_reference else "target_index"

    record_observation(
        scenario=f"p2_in_memory_{algorithm}_tool_list",
        perf_ids=("PERF-07",),
        fixture={
            "unrelated_records": 1000,
            "target_records": 1,
            "algorithm": algorithm,
        },
        samples_ms=(elapsed_ms,),
        counters=counters,
    )
