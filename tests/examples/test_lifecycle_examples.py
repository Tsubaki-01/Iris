from __future__ import annotations

from pathlib import Path

import pytest

from examples.lifecycle import events as events_example
from examples.lifecycle import start as start_example
from examples.lifecycle import status as status_example
from examples.lifecycle._runner import NonExecutingProvider
from iris.harness import (
    AgentRunner,
    AgentRunOptions,
    AgentRunRequest,
    RunLimits,
    RunResult,
    RuntimeExecutionOptions,
)
from iris.message import LLMRequest, LLMResponse, TextBlock
from iris.store import SQLiteStore


class RecordingRunner:
    def __init__(
        self,
        *,
        snapshot: object | None = None,
        result: object | None = None,
        events: list[object] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.result = result
        self.events = events or []
        self.start_call: tuple[AgentRunRequest, AgentRunOptions] | None = None
        self.status_calls: list[tuple[str, str]] = []
        self.events_call: tuple[str, int] | None = None
        self.resume_response: object | None = None
        self.cancel_call: tuple[str, str | None, float | None] | None = None
        self.recover_call: tuple[str, str] | None = None

    async def start(
        self,
        request: AgentRunRequest,
        *,
        options: AgentRunOptions,
    ) -> object:
        self.start_call = (request, options)
        return self.result

    def get_run(self, run_id: str) -> object:
        self.status_calls.append(("get_run", run_id))
        return self.snapshot

    def get_result(self, run_id: str) -> object | None:
        self.status_calls.append(("get_result", run_id))
        return self.result

    def list_events(
        self,
        run_id: str,
        after_sequence: int = 0,
        *,
        limit: int | None = None,
    ) -> list[object]:
        del limit
        self.events_call = (run_id, after_sequence)
        return self.events

    async def resume(
        self,
        run_id: str,
        *,
        interaction_id: str,
        response: object,
    ) -> object:
        del run_id, interaction_id
        self.resume_response = response
        return self.result

    async def cancel(
        self,
        run_id: str,
        *,
        reason: str | None,
        settlement_timeout: float | None,
    ) -> object:
        self.cancel_call = (run_id, reason, settlement_timeout)
        return self.result

    async def recover(
        self,
        run_id: str,
        *,
        expected_activation_id: str,
    ) -> object:
        self.recover_call = (run_id, expected_activation_id)
        return self.result


@pytest.mark.asyncio
async def test_start_run_maps_request_and_options() -> None:
    runner = RecordingRunner(result=object())
    result = await start_example.start_run(
        runner,  # type: ignore[arg-type]
        input_text="执行",
        session_id="session-1",
        run_id="run-1",
        max_steps=4,
        include_tools=False,
    )
    assert runner.start_call is not None
    request, options = runner.start_call
    assert result is runner.result
    assert request == AgentRunRequest(
        input="执行",
        session_id="session-1",
        run_id="run-1",
    )
    assert options == AgentRunOptions(
        limits=RunLimits(max_model_steps=4),
        runtime=RuntimeExecutionOptions(include_tools=False),
    )


class StaticProvider:
    def __init__(self, response: LLMResponse) -> None:
        self.response = response

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """为非计量测试返回固定输入估算。"""
        return 1

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        return self.response


@pytest.mark.asyncio
async def test_status_and_events_read_durable_sqlite_state(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "examples/lifecycle/agent.yaml"
    database = tmp_path / "lifecycle.db"
    writer = AgentRunner.from_config_path(
        config_path,
        provider=StaticProvider(
            LLMResponse(
                provider="fake",
                model="fake-model",
                content=[TextBlock(text="完成")],
                finish_reason="stop",
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
            )
        ),
        store=SQLiteStore(database),
    )
    await start_example.start_run(
        writer,
        input_text="读取持久化状态",
        session_id="session-1",
        run_id="run-1",
        max_steps=2,
        include_tools=False,
    )

    reader = AgentRunner.from_config_path(
        config_path,
        provider=NonExecutingProvider(),
        store=SQLiteStore(database),
    )
    subject = status_example.read_status(reader, run_id="run-1")
    assert isinstance(subject, RunResult)
    assert subject.run.run_id == "run-1"
    events, cursor = events_example.read_events(
        reader,
        run_id="run-1",
        after_sequence=0,
    )
    assert events
    assert cursor == events[-1].sequence
