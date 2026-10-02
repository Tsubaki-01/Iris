"""真实 YAML 子代理路由中的 Todo 开关、会话文件和恢复隔离。"""

from pathlib import Path

import pytest

from iris.exceptions import IrisRunPersistenceError
from iris.harness import AgentRunner
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import AgentRunOptions, AgentRunRequest, RunLimits, RunPhase, RunStopReason
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import SQLiteStore

from .fakes import StaticProvider, text_response, tool_response
from .test_runner_subagent import ChildProviders


def _write_configs(workspace: Path, parent_enabled: bool, child_enabled: bool | None) -> Path:
    parent_path = workspace / "agent.yaml"
    parent_path.write_text(
        "name: parent\nmodel: openai/test\nsystem: Parent instructions\n"
        "permissions:\n  workspace: .\n  writes: allow\n"
        "tools:\n  subagent: subagents.yaml\n"
        f"todo:\n  enabled: {str(parent_enabled).lower()}\n",
        encoding="utf-8",
    )
    (workspace / "subagents.yaml").write_text(
        "default: worker\nagents:\n  worker:\n    path: child.yaml\n"
        "    description: Handle the selected child task\n",
        encoding="utf-8",
    )
    (workspace / "child.yaml").write_text(
        "name: child\nmodel: openai/test\nsystem: Child instructions\n"
        "permissions:\n  workspace: .\n  writes: allow\n"
        + "tools:\n  builtin: "
        + ("[file.write, human.ask]\n" if child_enabled else "[human.ask]\n")
        + (
            f"todo:\n  enabled: {str(child_enabled).lower()}\n" if child_enabled is not None else ""
        ),
        encoding="utf-8",
    )
    return parent_path


def _snapshot(request: LLMRequest) -> str:
    return "\n".join(
        message.text
        for message in request.messages
        if message.metadata.get("context_kind") == "runtime_snapshot"
    )


def _parent_provider(enabled: bool) -> StaticProvider:
    return StaticProvider(
        tool_response(ToolUseBlock(id="delegate", name="subagent", input={"prompt": "处理子任务"})),
        text_response("父候选回复"),
        *([text_response("父自查后结束")] if enabled else []),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_enabled", [False, True])
@pytest.mark.parametrize("child_enabled", [None, False, True])
async def test_parent_child_yaml_switches_keep_independent_todo_files_and_reminders(
    tmp_path: Path, parent_enabled: bool, child_enabled: bool | None
) -> None:
    """显式和缺省开关由各自 YAML 决定，共享 workspace 也不继承或合并清单。"""
    config = _write_configs(tmp_path, parent_enabled, child_enabled)
    parent_path = tmp_path / ".iris/todos" / f"{b'parent-session'.hex()}.md"
    parent_path.parent.mkdir(parents=True)
    parent_content = "- [ ] parent-exclusive-work\n"
    parent_path.write_text(parent_content, encoding="utf-8")

    class ChildProvider(StaticProvider):
        """通过真实文件工具建立子清单，而非向 runtime 注入人为快照。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """从已建立的真实 child link 取身份，模型写自己的文件。"""
            self.requests.append(request)
            if child_enabled and len(self.requests) == 1:
                link = runner.store.load_subagent_link("parent", "delegate")
                child_run = runner.store.load_run(link.child_run_id)
                path = tmp_path / ".iris/todos" / f"{child_run.session_id.encode().hex()}.md"
                return tool_response(
                    ToolUseBlock(
                        id="child-plan",
                        name="write_file",
                        input={"file_path": str(path), "content": "- [ ] child-exclusive-work\n"},
                    )
                )
            return text_response("子任务需要等待，结束本轮")

    parent, child = _parent_provider(parent_enabled), ChildProvider()
    child_factory = ChildProviders(child)
    runner = AgentRunner.from_config_path(
        config, provider=parent, child_provider_factory=child_factory
    )
    try:
        result = await runner.start(
            AgentRunRequest(input="委派任务", session_id="parent-session", run_id="parent"),
            options=AgentRunOptions(limits=RunLimits(max_model_steps=6)),
        )
        assert result.run.stop_reason is RunStopReason.COMPLETED
        link = runner.store.load_subagent_link("parent", "delegate")
        child_run = runner.store.load_run(link.child_run_id)
        child_path = tmp_path / ".iris/todos" / f"{child_run.session_id.encode().hex()}.md"
        assert child_run.session_id != "parent-session"
        assert child_path != parent_path
        assert child_factory.configs[0][0].todo.enabled is bool(child_enabled)
        assert parent_path.read_text(encoding="utf-8") == parent_content
        assert sum("Todo 结束自查" in _snapshot(request) for request in parent.requests) == int(
            parent_enabled
        )
        assert sum("Todo 结束自查" in _snapshot(request) for request in child.requests) == int(
            bool(child_enabled)
        )
        for request in parent.requests:
            snapshot = _snapshot(request)
            assert ("[iris.todo]" in snapshot) is parent_enabled
            assert "child-exclusive-work" not in snapshot
            assert str(child_path) not in snapshot
            if parent_enabled:
                assert str(parent_path) in snapshot and "parent-exclusive-work" in snapshot
        for request in child.requests:
            snapshot = _snapshot(request)
            assert ("[iris.todo]" in snapshot) is bool(child_enabled)
            assert str(parent_path) not in snapshot and "parent-exclusive-work" not in snapshot
            if child_enabled:
                assert str(child_path) in snapshot
            else:
                assert all(tool.name != "write_file" for tool in request.tools)
        if child_enabled:
            assert child_path.read_text(encoding="utf-8") == "- [ ] child-exclusive-work\n"
            assert "child-exclusive-work" in _snapshot(child.requests[1])
            assert child_run.usage.model_steps_committed == 3
            if parent_enabled:
                assert (await runner.get_todo("parent-session")).items[0].content == (
                    "parent-exclusive-work"
                )
        else:
            assert not child_path.exists()
            assert child_run.usage.model_steps_committed == 1
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_waiting_child_recovers_original_session_todo_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实 question 等待后 child resume 中断，重启从原 child 身份读取人工更新文件。"""
    config = _write_configs(tmp_path, True, True)
    database = tmp_path / "child-recovery.db"

    class ChildPlanningProvider(StaticProvider):
        """先通过普通文件工具创建清单，再提出真实人工问题。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """首次请求创建自己文件，随后沿标准 question WAITING 路径。"""
            if not self.requests:
                self.requests.append(request)
                link = first.store.load_subagent_link("parent", "delegate")
                child_run = first.store.load_run(link.child_run_id)
                path = tmp_path / ".iris/todos" / f"{child_run.session_id.encode().hex()}.md"
                return tool_response(
                    ToolUseBlock(
                        id="plan",
                        name="write_file",
                        input={"file_path": str(path), "content": "- [ ] child-before-wait\n"},
                    )
                )
            return await super().complete(request)

    child = ChildPlanningProvider(
        tool_response(ToolUseBlock(id="ask", name="ask_question", input={"question": "继续吗？"}))
    )
    first = AgentRunner.from_config_path(
        config,
        provider=_parent_provider(True),
        store=SQLiteStore(database),
        child_provider_factory=ChildProviders(child),
    )
    try:
        parent_path = (await first.get_todo("parent-session")).path
        parent_path.parent.mkdir(parents=True)
        parent_path.write_text("- [ ] parent-stays-separate\n", encoding="utf-8")
        waiting = await first.start(
            AgentRunRequest(input="委派任务", session_id="parent-session", run_id="parent")
        )
        assert waiting.run.phase is RunPhase.WAITING
        proxy = waiting.pending_interaction
        child_id = proxy.request.subagent_origin.child_run_id
        child_before = first.store.load_run(child_id)
        assert child_before.phase is RunPhase.WAITING
        child_path = (await first.get_todo(child_before.session_id)).path
        child_path.write_text("- [-] child-human-change-before-recovery\n", encoding="utf-8")
        run_activation = AgentRunner._run_activation

        async def stop_child_resume(self: AgentRunner, active: object, **kwargs: object) -> object:
            if active.run_id == child_id and kwargs["activation"].kind == "resume":
                raise IrisRunPersistenceError("child stopped after resume admission")
            return await run_activation(self, active, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(AgentRunner, "_run_activation", stop_child_resume)
            with pytest.raises(IrisRunPersistenceError, match="child stopped"):
                await first.resume(
                    "parent",
                    interaction_id=proxy.interaction_id,
                    response=QuestionInteractionResponse(answer="继续等待"),
                )
        child_crashed = first.store.load_run(child_id)
        assert child_crashed.phase is RunPhase.ACTIVE
    finally:
        await first.aclose()

    resumed_child = StaticProvider(text_response("子候选回复"), text_response("子自查后结束"))
    resumed_parent = StaticProvider(text_response("父候选回复"), text_response("父自查后结束"))
    second = AgentRunner.from_config_path(
        config,
        provider=resumed_parent,
        store=SQLiteStore(database),
        child_provider_factory=ChildProviders(resumed_child),
    )
    try:
        result = await second.recover("parent")
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert second.store.load_subagent_link("parent", "delegate").child_run_id == child_id
        recovered_child = second.store.load_run(child_id)
        assert recovered_child.session_id == child_before.session_id
        assert recovered_child.phase is RunPhase.TERMINAL
        assert recovered_child.current_activation_id != child_crashed.current_activation_id
        assert len(resumed_child.requests) == len(resumed_parent.requests) == 2
        for request in resumed_child.requests:
            assert str(child_path) in _snapshot(request)
            assert "child-human-change-before-recovery" in _snapshot(request)
            assert "child-before-wait" not in _snapshot(request)
            assert "parent-stays-separate" not in _snapshot(request)
        assert "Todo 结束自查" not in _snapshot(resumed_child.requests[0])
        assert "Todo 结束自查" in _snapshot(resumed_child.requests[1])
        assert parent_path.read_text(encoding="utf-8") == "- [ ] parent-stays-separate\n"
        assert (
            child_path.read_text(encoding="utf-8") == "- [-] child-human-change-before-recovery\n"
        )
        assert {path.name for path in parent_path.parent.glob("*.md")} == {
            parent_path.name,
            child_path.name,
        }
    finally:
        await second.aclose()
