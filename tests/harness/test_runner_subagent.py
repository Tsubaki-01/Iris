"""真实 child Runner 与 shared-store 委派的确定性集成测试。"""

from datetime import timedelta
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.exceptions import IrisRunPersistenceError, IrisRunRecoveryError
from iris.harness import AgentRunner
from iris.hitl import make_call_fingerprint
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    CommitModelStep,
    CreateRun,
    FinishRun,
    ReserveModelStep,
    RunCheckpoint,
    RunErrorInfo,
    RunLimits,
    RunPhase,
    RunResult,
    RunStopReason,
    RuntimeExecutionOptions,
    RunToolCallRecord,
    RunUsage,
)
from iris.lifecycle.store import AdmitChildRun
from iris.message import Msg, ToolUseBlock
from iris.runtime import RuntimeCursor, RuntimeProvider
from iris.store import SQLiteStore
from iris.tools import PreparedToolCall, ToolResult
from iris.tools.permissions import DefaultPermissionPolicy, PermissionDecision, PermissionEffect
from iris.tools.subagent import ChildWaiting, SubagentExecutionOutcome

from .fakes import FrozenClock, StaticProvider, text_response, tool_response


def _write_configs(tmp_path: Path) -> Path:
    (tmp_path / "agent.yaml").write_text(
        "name: parent\nmodel: openai/parent-model\nsystem: Parent instructions\n"
        "tools:\n  subagent: subagents.yaml\n",
        encoding="utf-8",
    )
    (tmp_path / "subagents.yaml").write_text(
        "default: researcher\nagents:\n  researcher:\n    path: child.yaml\n"
        "    description: Research selected task\n  broken:\n    path: broken.yaml\n"
        "    description: Invalid unselected child\n",
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        "name: researcher\nmodel: openai/child-model\nsystem: Child instructions\n"
        "tools:\n  builtin: [file.read, human.ask]\n  subagent: missing-catalog.yaml\n"
        "session:\n  backend: sqlite\n  path: never-created.db\n",
        encoding="utf-8",
    )
    (tmp_path / "broken.yaml").write_text("[broken", encoding="utf-8")
    return tmp_path / "agent.yaml"


class ChildProviders:
    """记录按 selected child config 构造 provider 的实际输入。"""

    def __init__(self, provider: StaticProvider) -> None:
        self.provider = provider
        self.configs: list[tuple[AgentConfig, Path]] = []

    def __call__(self, config: AgentConfig, *, config_path: Path) -> RuntimeProvider:
        self.configs.append((config, config_path))
        return self.provider


def _prepare_parent(
    runner: AgentRunner,
    *,
    selector: str | None = None,
    options: AgentRunOptions | None = None,
) -> PreparedToolCall:
    """只提交真实 parent model/tool facts，phase04 不运行 parent tool loop。"""
    command, cursor = runner._build_start_facts(
        AgentRunRequest(
            input="Parent private request",
            session_id="parent-session",
            run_id="parent",
            metadata={"parent-only": True},
        ),
        options=options,
    )
    created = runner.store.create_run(command)
    reserved = runner.store.reserve_model_step(
        ReserveModelStep(
            run_id="parent",
            expected_run_revision=created.run.revision,
            activation_id=command.start_activation_id,
            now=runner.clock.now(),
        )
    )
    tool_use = ToolUseBlock(id="delegate", name="subagent", input={"prompt": "Child task"})
    if selector is not None:
        tool_use.input["agent"] = selector
    assistant = Msg.assistant([tool_use])
    prepared = runner.runtime.environment.tool_bridge.preflight_once(
        assistant_message=assistant,
        session_id="parent-session",
        run_id="parent",
        agent_id="parent",
        workspace_root=runner.runtime.environment.workspace_root,
        permission_mode="default",
        metadata=None,
    ).calls[0]
    after = RuntimeCursor(
        position="tool_batch", step_index=0, tool_calls=(tool_use,), assistant_message=assistant
    )
    fingerprint = make_call_fingerprint(
        session_id="parent-session",
        run_id="parent",
        tool_call_id="delegate",
        tool_name="subagent",
        arguments=prepared.arguments,
        workspace_root=str(runner.runtime.environment.workspace_root),
    )
    runner.store.commit_model_step(
        CommitModelStep(
            run_id="parent",
            expected_run_revision=reserved.run.revision,
            activation_id=command.start_activation_id,
            expected_session_revision=0,
            message_delta=[Msg.user("Parent private history"), assistant],
            usage=RunUsage(model_steps_reserved=1, model_steps_committed=1, total_tokens=11),
            prepared_tool_calls=[
                RunToolCallRecord(
                    run_id="parent",
                    step_index=0,
                    ordinal=1,
                    tool_call_id="delegate",
                    tool_name="subagent",
                    arguments=prepared.arguments,
                    fingerprint=fingerprint,
                    phase="prepared",
                    version=1,
                    created_at=runner.clock.now(),
                    updated_at=runner.clock.now(),
                )
            ],
            checkpoint=command.initial_checkpoint.model_copy(
                update={
                    "sequence": 2,
                    "session_revision": 1,
                    "model_steps_reserved": 1,
                    "model_steps_committed": 1,
                    "engine_cursor": after.model_dump(mode="json"),
                }
            ),
            assistant_message=assistant,
            now=runner.clock.now(),
        )
    )
    return prepared


async def _dispatch(
    runner: AgentRunner, prepared: PreparedToolCall, *, linked: bool = False
) -> SubagentExecutionOutcome:
    return await runner.runtime.environment.tool_bridge.execute_subagent_prepared(
        prepared,
        session_id="parent-session",
        run_id="parent",
        agent_id="parent",
        workspace_root=runner.runtime.environment.workspace_root,
        permission_mode="default",
        metadata=None,
        linked_continuation=linked,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", [None, "researcher"])
async def test_child_uses_fresh_context_options_and_shared_store(
    tmp_path: Path, selector: str | None
) -> None:
    path = _write_configs(tmp_path)
    (tmp_path / "notes.txt").write_text("local notes", encoding="utf-8")
    child_provider = StaticProvider(
        tool_response(
            ToolUseBlock(id="child-read", name="read_file", input={"file_path": "notes.txt"})
        ),
        text_response("Child answer"),
    )
    factory = ChildProviders(child_provider)
    parent_provider = StaticProvider()
    runner = AgentRunner.from_config_path(
        path, provider=parent_provider, child_provider_factory=factory
    )
    # 装配后的 catalog 不再读取；未选 child YAML 从未需要有效。
    (tmp_path / "subagents.yaml").write_text("[invalid now", encoding="utf-8")
    assert factory.configs == []
    options = AgentRunOptions(
        limits=RunLimits(max_model_steps=1),
        runtime=RuntimeExecutionOptions(
            request_options={"temperature": 0.1},
            memory_results=[{"parent-only": True}],
        ),
    )
    prepared = _prepare_parent(runner, selector=selector, options=options)
    result = await _dispatch(runner, prepared)
    assert isinstance(result, ToolResult)
    assert result.model_content == "Child answer"
    child_id = result.metadata["child_run_id"]
    child = runner.store.load_run(child_id)
    assert child.request.input == "Child task"
    assert child.request.metadata == {}
    assert child.options == AgentRunOptions()
    assert child.session_id != "parent-session"
    assert child.run_id != "parent"
    assert child.phase == RunPhase.TERMINAL
    assert child.usage.tool_calls_committed == 1
    assert result.metadata == {"agent_selector": "researcher", "child_run_id": child_id}
    assert runner.store.load_run("parent").usage.total_tokens == 11
    assert runner.store.load_run("parent").usage.tool_calls_committed == 0
    assert runner.store.load_tool_call("parent", "delegate").phase.value == "prepared"
    assert parent_provider.requests == []
    assert len(factory.configs) == 1
    assert factory.configs[0][0].model.name == "child-model"
    assert factory.configs[0][1] == tmp_path / "child.yaml"
    request = child_provider.requests[0]
    texts = "\n".join(message.text for message in request.messages)
    assert "Child instructions" in texts and "Child task" in texts
    assert "Parent private" not in texts and "Parent instructions" not in texts
    assert not (tmp_path / "never-created.db").exists()
    assert await _dispatch(runner, prepared, linked=True) == result
    assert len(factory.configs) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,code", [("broken", "SUBAGENT_CONFIG_ERROR"), ("disjoint", "SUBAGENT_WORKSPACE_DISJOINT")]
)
async def test_fresh_selected_config_failures_do_not_admit_child(
    tmp_path: Path, kind: str, code: str
) -> None:
    path = _write_configs(tmp_path)
    if kind == "disjoint":
        with (tmp_path / "child.yaml").open("a", encoding="utf-8") as config:
            config.write("permissions:\n  workspace: ../outside\n")
    runner = AgentRunner.from_config_path(
        path, provider=StaticProvider(), child_provider_factory=ChildProviders(StaticProvider())
    )
    prepared = _prepare_parent(runner, selector="broken" if kind == "broken" else None)
    result = await _dispatch(runner, prepared)
    assert result.error.code == code
    assert not result.error.retryable
    assert result.metadata == {"agent_selector": "broken" if kind == "broken" else "researcher"}
    assert runner.store.load_subagent_link("parent", "delegate") is None


@pytest.mark.asyncio
async def test_waiting_child_reentry_reads_original_result_without_yaml_or_provider(
    tmp_path: Path,
) -> None:
    path = _write_configs(tmp_path)
    factory = ChildProviders(
        StaticProvider(
            tool_response(
                ToolUseBlock(
                    id="ask",
                    name="ask_question",
                    input={"question": "Which notes?"},
                )
            )
        )
    )
    clock = FrozenClock()
    runner = AgentRunner.from_config_path(
        path, provider=StaticProvider(), child_provider_factory=factory, clock=clock
    )
    prepared = _prepare_parent(
        runner,
        options=AgentRunOptions(
            runtime=RuntimeExecutionOptions(tool_timeout_seconds=10),
            limits=RunLimits(interaction_timeout_seconds=30),
        ),
    )
    waiting = await _dispatch(runner, prepared)
    assert isinstance(waiting, ChildWaiting)
    assert waiting.expiry_owner.value == "outer_tool_timeout"
    assert waiting.proxy_expires_at == clock.now() + timedelta(seconds=10)
    (tmp_path / "child.yaml").write_text("[broken after admission", encoding="utf-8")
    clock.advance(seconds=2)
    again = await _dispatch(runner, prepared, linked=True)
    assert again == waiting
    assert len(factory.configs) == 1
    assert runner.store.load_run("parent").phase == RunPhase.ACTIVE


def _admit_durable_child(runner: AgentRunner) -> str:
    parent = runner.store.load_run("parent")
    child_create = CreateRun(
        request=AgentRunRequest(input="Child task", session_id="child-session", run_id="child"),
        options=AgentRunOptions(),
        agent_id="researcher",
        environment_fingerprint="child-env",
        start_activation_id="child-a",
        initial_checkpoint=RunCheckpoint(
            run_id="child",
            sequence=1,
            activation_id="child-a",
            engine_cursor={},
            session_revision=0,
            model_steps_reserved=0,
            model_steps_committed=0,
            environment_fingerprint="child-env",
        ),
        now=runner.clock.now(),
    )
    return runner.store.admit_child_run(
        AdmitChildRun(
            parent_run_id="parent",
            expected_parent_run_revision=parent.revision,
            parent_activation_id=parent.current_activation_id,
            parent_tool_call_id="delegate",
            expected_parent_tool_version=1,
            child_create=child_create,
        )
    ).child_run_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,code",
    [
        (RunStopReason.FAILED, "SUBAGENT_FAILED"),
        (RunStopReason.BUDGET_EXHAUSTED, "SUBAGENT_BUDGET_EXHAUSTED"),
        (RunStopReason.DEADLINE_EXCEEDED, "SUBAGENT_TIMEOUT"),
        (RunStopReason.INTERACTION_EXPIRED, "SUBAGENT_TIMEOUT"),
        (RunStopReason.CANCELLED, "SUBAGENT_CANCELLED"),
        (RunStopReason.OUTCOME_UNKNOWN, "SUBAGENT_OUTCOME_UNKNOWN"),
        (RunStopReason.COMPLETED, "SUBAGENT_FAILED"),
    ],
)
async def test_linked_terminal_projection_is_nonretryable_and_run_local(
    tmp_path: Path,
    reason: RunStopReason,
    code: str,
) -> None:
    runner = AgentRunner.from_config_path(_write_configs(tmp_path), provider=StaticProvider())
    prepared = _prepare_parent(runner)
    child_id = _admit_durable_child(runner)
    runner.store.finish_run(
        FinishRun(
            run_id=child_id,
            expected_run_revision=1,
            activation_id="child-a",
            stop_reason=reason,
            error=RunErrorInfo(code="CHILD_FAILURE", message="details", source="runtime")
            if reason in {RunStopReason.FAILED, RunStopReason.OUTCOME_UNKNOWN}
            else None,
            now=runner.clock.now(),
        )
    )
    result = await _dispatch(runner, prepared, linked=True)
    assert result.error.code == code
    assert not result.error.retryable
    assert result.content == [] and result.data == {} and result.stats == {}
    assert result.metadata == {"agent_selector": "researcher", "child_run_id": child_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", [False, True])
async def test_restart_recovers_original_active_child_with_ordinary_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: bool,
) -> None:
    path = _write_configs(tmp_path)
    db = tmp_path / "state.db"
    factory = ChildProviders(StaticProvider(text_response("Recovered child")))
    runner = AgentRunner.from_config_path(
        path, provider=StaticProvider(), store=SQLiteStore(db), child_provider_factory=factory
    )
    prepared = _prepare_parent(runner)

    async def stop_after_admission(
        self: AgentRunner, *, run_id: str, activation_id: str
    ) -> RunResult:
        raise IrisRunPersistenceError("process stopped after admission")

    with monkeypatch.context() as patch:
        patch.setattr(AgentRunner, "_run_admitted_start", stop_after_admission)
        with pytest.raises(IrisRunPersistenceError):
            await _dispatch(runner, prepared)
    child_id = runner.store.load_subagent_link("parent", "delegate").child_run_id
    if drift:
        child_path = tmp_path / "child.yaml"
        child_path.write_text(
            child_path.read_text(encoding="utf-8").replace(
                "Child instructions", "Changed instructions"
            ),
            encoding="utf-8",
        )
    restarted = AgentRunner.from_config_path(
        path, provider=StaticProvider(), store=SQLiteStore(db), child_provider_factory=factory
    )
    durable = restarted.store.load_tool_call("parent", "delegate")
    rebuilt = restarted.runtime.environment.tool_bridge.prepare_subagent_continuation(
        ToolUseBlock(id=durable.tool_call_id, name=durable.tool_name, input=durable.arguments),
        session_id="parent-session",
        run_id="parent",
        agent_id="parent",
        workspace_root=tmp_path,
        permission_mode="default",
        metadata=None,
    )
    if drift:
        with pytest.raises(IrisRunRecoveryError):
            await _dispatch(restarted, rebuilt, linked=True)
    else:
        result = await _dispatch(restarted, rebuilt, linked=True)
        assert result.model_content == "Recovered child"
        assert result.metadata["child_run_id"] == child_id
        assert restarted.store.load_run(child_id).phase == RunPhase.TERMINAL
    assert restarted.store.load_subagent_link("parent", "delegate").child_run_id == child_id
    assert (
        len(
            [
                event
                for event in restarted.store.list_events(child_id)
                if event.kind.value == "run.started"
            ]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_success_keeps_empty_assistant_text(tmp_path: Path) -> None:
    factory = ChildProviders(StaticProvider(text_response("")))
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path), provider=StaticProvider(), child_provider_factory=factory
    )
    result = await _dispatch(runner, _prepare_parent(runner))
    assert not result.is_error
    assert result.model_content == ""


@pytest.mark.asyncio
async def test_runner_injected_parent_policy_applies_to_child_only_tool(tmp_path: Path) -> None:
    class ParentPolicy(DefaultPermissionPolicy):
        def check(self, tool: object, params: object, context: object) -> PermissionDecision:
            if tool.name == "read_file":
                return PermissionDecision(
                    effect=PermissionEffect.DENY, reason="parent denies reading"
                )
            return PermissionDecision(effect=PermissionEffect.ALLOW)

        def fingerprint_payload(self) -> dict[str, object]:
            return {"type": "parent_read_denial"}

    provider = StaticProvider(
        tool_response(
            ToolUseBlock(
                id="read",
                name="read_file",
                input={"file_path": "notes.txt"},
            )
        ),
        text_response("Cannot read"),
    )
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=StaticProvider(),
        permission_policy=ParentPolicy(),
        child_provider_factory=ChildProviders(provider),
    )
    result = await _dispatch(runner, _prepare_parent(runner))
    assert result.model_content == "Cannot read"
    child_result = runner.store.load_tool_call(result.metadata["child_run_id"], "read").result
    assert child_result.error.code == "PERMISSION_ERROR"
    assert child_result.error.message == "parent denies reading"


def test_runner_reads_catalog_once_and_defers_child_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_configs(tmp_path)
    read_text = Path.read_text
    reads: list[Path] = []

    def record_read(self: Path, *args: object, **kwargs: object) -> str:
        reads.append(self)
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_read)
    AgentRunner.from_config_path(path, provider=StaticProvider())
    assert reads.count(tmp_path / "subagents.yaml") == 1
    assert tmp_path / "child.yaml" not in reads
    assert tmp_path / "broken.yaml" not in reads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate,owner",
    [
        ("none", None),
        ("parent_deadline", "parent_run_deadline"),
        ("parent_timeout", "parent_interaction_timeout"),
        ("child_interaction", "child_interaction_expiry"),
        ("child_deadline", "child_effective_deadline"),
        ("outer", "outer_tool_timeout"),
        ("tie", "parent_run_deadline"),
    ],
)
async def test_waiting_expiry_projection_uses_earliest_absolute_candidate(
    tmp_path: Path,
    candidate: str,
    owner: str | None,
) -> None:
    clock = FrozenClock()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=StaticProvider(),
        clock=clock,
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(
                    ToolUseBlock(
                        id="ask",
                        name="ask_question",
                        input={"question": "Continue?"},
                    )
                )
            )
        ),
    )
    waiting = await _dispatch(runner, _prepare_parent(runner))
    parent = runner.store.load_run("parent")
    child = runner.store.load_run(waiting.child_run_id)
    result = runner.store.load_result(child.run_id)
    soon = clock.now() + timedelta(seconds=5)
    limits = RunLimits(
        deadline_at=soon if candidate in {"parent_deadline", "tie"} else None,
        interaction_timeout_seconds=5 if candidate in {"parent_timeout", "tie"} else None,
    )
    parent = parent.model_copy(
        update={
            "options": AgentRunOptions(
                limits=limits,
                runtime=RuntimeExecutionOptions(
                    tool_timeout_seconds=5 if candidate in {"outer", "tie"} else None
                ),
            )
        }
    )
    if candidate == "child_deadline":
        child = child.model_copy(
            update={"options": AgentRunOptions(limits=RunLimits(deadline_at=soon))}
        )
    if candidate == "child_interaction":
        result = result.model_copy(
            update={
                "pending_interaction": result.pending_interaction.model_copy(
                    update={"expires_at": soon}
                )
            }
        )
    projected = runner._subagent_controller._project_outcome("researcher", parent, child, result)
    assert projected.expiry_owner == owner
    assert projected.proxy_expires_at == (None if owner is None else soon)
