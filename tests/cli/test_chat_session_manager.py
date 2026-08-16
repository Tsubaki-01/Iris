from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from iris.cli.chat import ChatOptions, run_chat_loop
from iris.harness import AgentRunner, RunEvent, RunEventKind, RunStopReason
from iris.message import LLMRequest, LLMResponse
from iris.store import InMemoryLifecycleStore
from tests.harness.fakes import build_runtime, text_response


class DelayedProvider:
    """延迟返回确定性文本，给 CLI 留出 busy 输入窗口。"""

    def __init__(self, *, delay: float) -> None:
        self.delay = delay
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """记录请求，并按调用序号返回不同文本。"""
        self.requests.append(request)
        index = len(self.requests)
        await asyncio.sleep(self.delay)
        return text_response(f"完成-{index}")


class RecordingObserver:
    """记录 runner 已完成投递的 durable events。"""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def on_event(self, event: RunEvent) -> None:
        """保存一条 durable event。"""
        self.events.append(event)


def _started_run_ids(observer: RecordingObserver) -> list[str]:
    """返回 observer 看到的 run.started 对应 run id。"""
    return [event.run_id for event in observer.events if event.kind is RunEventKind.RUN_STARTED]


def test_busy_plain_text_steers_current_run(tmp_path: Path) -> None:
    """Busy 普通文本必须进入当前 run，而不是等 terminal 后新建 run。"""
    provider = DelayedProvider(delay=0.02)
    store = InMemoryLifecycleStore()
    observer = RecordingObserver()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=store,
        observers=[observer],
    )
    finished = threading.Event()
    outputs: list[str] = []
    errors: list[str] = []
    input_index = 0

    def read_input(prompt: str) -> str:
        nonlocal input_index
        del prompt
        input_index += 1
        if input_index == 1:
            return "开始"
        if input_index == 2:
            return "调整方向"
        assert finished.wait(1)
        return "/exit"

    def write_output(message: str) -> None:
        outputs.append(message)
        if message == "完成-2":
            finished.set()

    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=read_input,
        output_func=write_output,
        error_func=errors.append,
    )

    assert code == 0
    assert len(set(_started_run_ids(observer))) == 1
    assert [message.text for message in store.load_session("cli").messages] == [
        "开始",
        "完成-1",
        "调整方向",
        "完成-2",
    ]
    assert errors == []


def test_follow_up_command_queues_next_run(tmp_path: Path) -> None:
    """Busy /follow-up 必须等当前 run terminal 后创建下一 run。"""
    provider = DelayedProvider(delay=0.02)
    store = InMemoryLifecycleStore()
    observer = RecordingObserver()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=store,
        observers=[observer],
    )
    finished = threading.Event()
    outputs: list[str] = []
    errors: list[str] = []
    input_index = 0

    def read_input(prompt: str) -> str:
        nonlocal input_index
        del prompt
        input_index += 1
        if input_index == 1:
            return "第一轮"
        if input_index == 2:
            return "/follow-up 第二轮"
        finished.wait(1)
        return "/exit"

    def write_output(message: str) -> None:
        outputs.append(message)
        if message == "完成-2":
            finished.set()

    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=read_input,
        output_func=write_output,
        error_func=errors.append,
    )

    assert code == 0
    assert len(set(_started_run_ids(observer))) == 2
    assert [message.text for message in store.load_session("cli").messages] == [
        "第一轮",
        "完成-1",
        "第二轮",
        "完成-2",
    ]
    assert errors == []


def test_keyboard_interrupt_interrupts_active_run_before_exit(tmp_path: Path) -> None:
    """Active Ctrl-C 必须先写入 cancellation request，再保持退出码 130。"""
    provider = DelayedProvider(delay=0.2)
    store = InMemoryLifecycleStore()
    observer = RecordingObserver()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider),
        store=store,
        observers=[observer],
    )
    errors: list[str] = []
    input_index = 0

    def read_input(prompt: str) -> str:
        nonlocal input_index
        del prompt
        input_index += 1
        if input_index == 1:
            return "开始"
        raise KeyboardInterrupt

    code = run_chat_loop(
        runner=runner,
        options=ChatOptions(config_path=tmp_path / "agent.yaml"),
        input_func=read_input,
        output_func=lambda message: None,
        error_func=errors.append,
    )

    started_run_ids = _started_run_ids(observer)
    assert code == 130
    assert len(set(started_run_ids)) == 1
    run = runner.get_run(started_run_ids[0])
    assert run.cancellation_requested_at is not None
    assert run.stop_reason is RunStopReason.CANCELLED
    assert errors == []
