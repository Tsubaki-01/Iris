from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from rich.console import Console

import iris.runtime.runtime as runtime_module
from iris.agents import AgentConfig
from iris.cli.chat import ChatOptions, run_chat_loop
from iris.cli.render import ChatRenderer
from iris.cli.trace import ChatTraceStore, TracingRuntimeProvider
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.hitl import (
    HumanInteraction,
    InteractionResumePhase,
    InteractionStatus,
    QuestionInteractionResponse,
)
from iris.message import LLMRequest, LLMResponse, TextBlock, ToolUseBlock
from iris.runtime import AgentRuntime
from iris.runtime.models import RuntimeOptions, RuntimeStatus, RuntimeTurnResult
from iris.store import SQLiteStore
from iris.tools import (
    AskQuestionTool,
    DefaultPermissionPolicy,
    ToolCapability,
    ToolExecutor,
    ToolRegistry,
)

SESSION_ID = "restart-session"


class RecordingProvider:
    """按顺序返回响应，并保留收到的请求。"""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("provider 收到计划外请求")
        return self.responses.pop(0)


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="recovery-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是恢复测试助手。",
    )


def _context_input() -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(slots=[ContextSlot(name="instructions", content="保持确定性")])
    )


def _tool_response(*calls: ToolUseBlock) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        model="gpt-4o-mini",
        content=[TextBlock(text="需要工具"), *calls],
        finish_reason="tool_calls",
    )


def _text_response(text: str = "恢复完成") -> LLMResponse:
    return LLMResponse(
        provider="fake",
        model="gpt-4o-mini",
        content=[TextBlock(text=text)],
        finish_reason="stop",
    )


def _make_runtime(
    database: Path,
    responses: list[LLMResponse],
    *,
    trace_store: ChatTraceStore | None = None,
    writes: list[str] | None = None,
    reads: list[str] | None = None,
) -> tuple[AgentRuntime, RecordingProvider, SQLiteStore]:
    write_calls = writes if writes is not None else []
    read_calls = reads if reads is not None else []

    def write_probe(value: str) -> str:
        write_calls.append(value)
        return f"written:{value}"

    def read_probe() -> str:
        read_calls.append("read")
        return "read"

    registry = ToolRegistry()
    registry.register_function(
        write_probe,
        description="写入恢复探针",
        capabilities={ToolCapability.WRITE},
    )
    registry.register_function(
        read_probe,
        description="读取恢复探针",
        capabilities={ToolCapability.READ},
    )
    registry.register(AskQuestionTool())
    delegate = RecordingProvider(responses)
    provider = (
        TracingRuntimeProvider(delegate, trace_store) if trace_store is not None else delegate
    )
    store = SQLiteStore(database)
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=store,
        interaction_store=store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(
            registry,
            permission_policy=DefaultPermissionPolicy(write_mode="confirm"),
        ),
        workspace_root=Path.cwd(),
    )
    return runtime, delegate, store


def _persist_waiting(runtime: AgentRuntime, response: LLMResponse) -> HumanInteraction:
    provider = runtime.provider
    assert isinstance(provider, RecordingProvider)
    provider.responses.append(response)
    waiting = asyncio.run(
        runtime.run_loop(
            "开始",
            options=RuntimeOptions(session_id=SESSION_ID, run_id="run-restart"),
        )
    )
    assert waiting.status is RuntimeStatus.WAITING_HUMAN
    assert waiting.pending_interaction is not None
    return waiting.pending_interaction


def _run_cli(
    runtime: AgentRuntime,
    trace_store: ChatTraceStore,
    inputs: list[str],
) -> tuple[int, str]:
    console = Console(record=True, width=120, color_system=None)
    tokens = iter(inputs)
    code = run_chat_loop(
        runtime=runtime,
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id=SESSION_ID),
        trace_store=trace_store,
        renderer=ChatRenderer(console),
        input_func=lambda prompt: next(tokens),
    )
    return code, console.export_text()


def _tool_result_contents(request: LLMRequest) -> list[str]:
    return [block.content for message in request.messages for block in message.tool_results]


def _tool_result_count(store: SQLiteStore, tool_use_id: str) -> int:
    count = 0
    for message in store.load_messages(SESSION_ID):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        count += sum(
            block.get("tool_use_id") == tool_use_id
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_result"
        )
    return count


def _event_count(store: SQLiteStore, tool_use_id: str) -> int:
    return sum(
        event.get("tool_call_id") == tool_use_id for event in store.load_tool_events(SESSION_ID)
    )


def test_fresh_session_starts_normal_turn_at_one(tmp_path: Path) -> None:
    trace_store = ChatTraceStore()
    runtime, provider, _ = _make_runtime(
        tmp_path / "session.db",
        [_text_response("普通答复")],
        trace_store=trace_store,
    )

    code, output = _run_cli(runtime, trace_store, ["你好", "/exit"])

    assert code == 0
    assert len(provider.requests) == 1
    assert trace_store.steps_for_turn(0) == []
    assert len(trace_store.steps_for_turn(1)) == 1
    assert "RECOVERY" not in output
    assert "USER #1" in output


def test_pending_permission_recovers_once_and_terminal_run_is_not_recovered_again(
    tmp_path: Path,
) -> None:
    database = tmp_path / "session.db"
    writes: list[str] = []
    first, _, _ = _make_runtime(database, [], writes=writes)
    interaction = _persist_waiting(
        first,
        _tool_response(
            ToolUseBlock(
                id="write",
                name="write_probe",
                input={"value": "once"},
            )
        ),
    )
    trace_store = ChatTraceStore()
    restarted, provider, store = _make_runtime(
        database,
        [_text_response()],
        trace_store=trace_store,
        writes=writes,
    )

    code, output = _run_cli(restarted, trace_store, ["y", "/exit"])

    assert code == 0
    assert writes == ["once"]
    assert len(provider.requests) == 1
    assert len(trace_store.steps_for_turn(0)) == 1
    assert interaction.interaction_id in output
    assert "RECOVERY" in output
    assert "PERMISSION" in output
    assert restarted.load_resumable_interaction(SESSION_ID) is None

    third, _, _ = _make_runtime(database, [], writes=writes)
    third_code, third_output = _run_cli(third, ChatTraceStore(), ["/exit"])
    assert third_code == 0
    assert "RECOVERY" not in third_output
    assert store.load_interaction(interaction.interaction_id) is not None


def test_pending_question_recovers_answer_into_provider_turn_zero(tmp_path: Path) -> None:
    database = tmp_path / "session.db"
    first, _, _ = _make_runtime(database, [])
    interaction = _persist_waiting(
        first,
        _tool_response(
            ToolUseBlock(
                id="question",
                name="ask_question",
                input={"question": "部署到哪里？", "options": ["测试", "生产"]},
            )
        ),
    )
    trace_store = ChatTraceStore()
    restarted, provider, _ = _make_runtime(
        database,
        [_text_response()],
        trace_store=trace_store,
    )

    code, output = _run_cli(restarted, trace_store, ["2", "/exit"])

    assert code == 0
    assert len(provider.requests) == 1
    assert "生产" in _tool_result_contents(provider.requests[0])
    assert len(trace_store.steps_for_turn(0)) == 1
    assert interaction.interaction_id in output
    assert "QUESTION" in output


def test_resolved_interaction_recovers_without_prompting_again(tmp_path: Path) -> None:
    database = tmp_path / "session.db"
    first, _, _ = _make_runtime(database, [])
    interaction = _persist_waiting(
        first,
        _tool_response(
            ToolUseBlock(
                id="question",
                name="ask_question",
                input={"question": "继续？"},
            )
        ),
    )
    first.interaction_service.resolve(
        interaction.interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )
    restarted, provider, _ = _make_runtime(database, [_text_response()])

    code, output = _run_cli(restarted, ChatTraceStore(), ["/exit"])

    assert code == 0
    assert len(provider.requests) == 1
    assert "继续" in _tool_result_contents(provider.requests[0])
    assert "RECOVERY" in output
    assert "QUESTION" not in output


def test_result_ready_recovery_commits_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "session.db"
    first, _, store = _make_runtime(database, [])
    interaction = _persist_waiting(
        first,
        _tool_response(
            ToolUseBlock(
                id="question",
                name="ask_question",
                input={"question": "继续？"},
            )
        ),
    )

    async def fail_before_commit(**_: object) -> RuntimeTurnResult:
        raise RuntimeError("模拟 result_ready 后进程退出")

    with monkeypatch.context() as crash_patch:
        crash_patch.setattr(runtime_module, "commit_ready_interaction", fail_before_commit)
        crash_patch.setattr(
            runtime_module,
            "synchronize_resume_metadata",
            lambda *, session_store, result: result,
        )
        interrupted = asyncio.run(
            first.resume(
                interaction.interaction_id,
                QuestionInteractionResponse(answer="继续"),
            )
        )
    assert interrupted.status is RuntimeStatus.ERROR
    ready = store.load_interaction(interaction.interaction_id)
    assert ready is not None
    assert ready.resume_phase is InteractionResumePhase.RESULT_READY

    restarted, provider, recovered_store = _make_runtime(database, [_text_response()])
    code, output = _run_cli(restarted, ChatTraceStore(), ["/exit"])

    assert code == 0
    assert len(provider.requests) == 1
    assert _tool_result_count(recovered_store, "question") == 1
    assert _event_count(recovered_store, "question") == 1
    assert "QUESTION" not in output
    assert restarted.load_resumable_interaction(SESSION_ID) is None


def test_result_committed_recovery_continues_without_replaying_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "session.db"
    reads: list[str] = []
    first, _, store = _make_runtime(database, [], reads=reads)
    interaction = _persist_waiting(
        first,
        _tool_response(
            ToolUseBlock(
                id="question",
                name="ask_question",
                input={"question": "继续？"},
            ),
            ToolUseBlock(id="read", name="read_probe", input={}),
        ),
    )

    async def fail_after_commit(**_: object) -> RuntimeTurnResult:
        raise RuntimeError("模拟 result_committed 后进程退出")

    with monkeypatch.context() as crash_patch:
        crash_patch.setattr(first, "_resume_batch", fail_after_commit)
        crash_patch.setattr(
            runtime_module,
            "synchronize_resume_metadata",
            lambda *, session_store, result: result,
        )
        interrupted = asyncio.run(
            first.resume(
                interaction.interaction_id,
                QuestionInteractionResponse(answer="继续"),
            )
        )
    assert interrupted.status is RuntimeStatus.ERROR
    committed = store.load_interaction(interaction.interaction_id)
    assert committed is not None
    assert committed.resume_phase is InteractionResumePhase.RESULT_COMMITTED
    assert reads == []

    restarted, provider, recovered_store = _make_runtime(
        database,
        [_text_response()],
        reads=reads,
    )
    code, output = _run_cli(restarted, ChatTraceStore(), ["/exit"])

    assert code == 0
    assert len(provider.requests) == 1
    assert reads == ["read"]
    assert _tool_result_count(recovered_store, "question") == 1
    assert _tool_result_count(recovered_store, "read") == 1
    assert _event_count(recovered_store, "question") == 1
    assert _event_count(recovered_store, "read") == 1
    assert "QUESTION" not in output


def test_claimed_unknown_outcome_fails_closed_without_reading_input(tmp_path: Path) -> None:
    database = tmp_path / "session.db"
    first, _, _ = _make_runtime(database, [])
    interaction = _persist_waiting(
        first,
        _tool_response(
            ToolUseBlock(
                id="question",
                name="ask_question",
                input={"question": "继续？"},
            )
        ),
    )
    first.interaction_service.resolve(
        interaction.interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )
    first.interaction_service.claim(interaction.interaction_id, interaction.checkpoint)
    restarted, provider, _ = _make_runtime(database, [])
    console = Console(record=True, width=120, color_system=None)

    def fail_on_input(prompt: str) -> str:
        raise AssertionError(f"不应读取普通输入: {prompt}")

    code = run_chat_loop(
        runtime=restarted,
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id=SESSION_ID),
        trace_store=ChatTraceStore(),
        renderer=ChatRenderer(console),
        input_func=fail_on_input,
    )

    assert code == 1
    assert provider.requests == []
    assert "HITL_EXECUTION_OUTCOME_UNKNOWN" in console.export_text()


def test_continuation_claim_fails_closed_without_replaying_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    database = tmp_path / "session.db"
    reads: list[str] = []
    first, _, store = _make_runtime(database, [], reads=reads)
    interaction = _persist_waiting(
        first,
        _tool_response(
            ToolUseBlock(
                id="question",
                name="ask_question",
                input={"question": "继续？"},
            ),
            ToolUseBlock(id="read", name="read_probe", input={}),
        ),
    )

    def fail_after_tool(**_: object) -> object:
        raise SimulatedProcessCrash

    with monkeypatch.context() as crash_patch:
        crash_patch.setattr(runtime_module, "append_resumed_result", fail_after_tool)
        with pytest.raises(SimulatedProcessCrash):
            asyncio.run(
                first.resume(
                    interaction.interaction_id,
                    QuestionInteractionResponse(answer="继续"),
                )
            )
    assert reads == ["read"]
    claimed = store.load_interaction(interaction.interaction_id)
    assert claimed is not None
    assert claimed.checkpoint["continuation_claim"]["tool_call_id"] == "read"

    restarted, provider, _ = _make_runtime(database, [], reads=reads)
    console = Console(record=True, width=120, color_system=None)

    def fail_on_input(prompt: str) -> str:
        raise AssertionError(f"不应读取普通输入: {prompt}")

    code = run_chat_loop(
        runtime=restarted,
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id=SESSION_ID),
        trace_store=ChatTraceStore(),
        renderer=ChatRenderer(console),
        input_func=fail_on_input,
    )

    assert code == 1
    assert provider.requests == []
    assert reads == ["read"]
    assert "HITL_EXECUTION_OUTCOME_UNKNOWN" in console.export_text()


@pytest.mark.parametrize("stale_marker", [False, True])
def test_followup_gate_is_recovered_by_marker_or_pending_fallback(
    tmp_path: Path,
    stale_marker: bool,
) -> None:
    database = tmp_path / "session.db"
    first, _, store = _make_runtime(database, [])
    first_interaction = _persist_waiting(
        first,
        _tool_response(
            ToolUseBlock(
                id="first",
                name="ask_question",
                input={"question": "第一问？"},
            ),
            ToolUseBlock(
                id="second",
                name="ask_question",
                input={"question": "第二问？", "options": ["甲", "乙"]},
            ),
        ),
    )
    second_waiting = asyncio.run(
        first.resume(
            first_interaction.interaction_id,
            QuestionInteractionResponse(answer="第一答"),
        )
    )
    assert second_waiting.pending_interaction is not None
    second_interaction = second_waiting.pending_interaction
    if stale_marker:
        metadata = store.load_run_metadata(SESSION_ID)
        raw_latest = metadata["latest_run"]
        assert isinstance(raw_latest, dict)
        latest = dict(raw_latest)
        latest["interaction_id"] = first_interaction.interaction_id
        metadata["latest_run"] = latest
        store.save_run_metadata(SESSION_ID, metadata)

    restarted, provider, _ = _make_runtime(database, [_text_response()])
    code, output = _run_cli(restarted, ChatTraceStore(), ["2", "/exit"])

    assert code == 0
    assert len(provider.requests) == 1
    assert "乙" in _tool_result_contents(provider.requests[0])
    assert second_interaction.interaction_id in output
    assert first_interaction.interaction_id not in output


@pytest.mark.parametrize(
    ("error_type", "expected_code"),
    [(KeyboardInterrupt, 130), (EOFError, 0)],
)
def test_startup_interaction_interrupt_remains_recoverable(
    tmp_path: Path,
    error_type: type[BaseException],
    expected_code: int,
) -> None:
    database = tmp_path / "session.db"
    first, _, _ = _make_runtime(database, [])
    interaction = _persist_waiting(
        first,
        _tool_response(
            ToolUseBlock(
                id="write",
                name="write_probe",
                input={"value": "later"},
            )
        ),
    )
    restarted, provider, store = _make_runtime(database, [])
    console = Console(record=True, width=120, color_system=None)

    def interrupt(prompt: str) -> str:
        raise error_type

    code = run_chat_loop(
        runtime=restarted,
        agent_config=_agent_config(),
        options=ChatOptions(config_path=Path("agent.yaml"), session_id=SESSION_ID),
        trace_store=ChatTraceStore(),
        renderer=ChatRenderer(console),
        input_func=interrupt,
    )

    assert code == expected_code
    assert provider.requests == []
    persisted = store.load_interaction(interaction.interaction_id)
    assert persisted is not None
    assert persisted.status is InteractionStatus.PENDING
    next_runtime, _, _ = _make_runtime(database, [])
    assert next_runtime.load_resumable_interaction(SESSION_ID) == persisted
    output = console.export_text()
    assert "RECOVERY" in output
    assert interaction.interaction_id in output
    assert "ASSISTANT" not in output
