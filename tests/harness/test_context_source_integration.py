"""通过真实 Runner 验证宿主动态状态与归档输入的边界。"""

import asyncio
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.context import (
    ContextBuildScope,
    ContextContribution,
    ContextSection,
    ContextSlot,
    ContextSnapshot,
)
from iris.harness import AgentRunner, AgentRunRequest
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .fakes import StaticProvider, text_response, tool_response
from .test_runner_subagent import ChildProviders, _write_configs


class _DocumentSource:
    """宿主自行选择当前文档状态；框架只接收明确的必需贡献。"""

    def __init__(self) -> None:
        self.current = "当前打开报告 A，选中第二段"
        self.scopes: list[ContextBuildScope] = []

    async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
        """返回本步骤完整快照，空状态不带上次的文档。"""
        self.scopes.append(scope)
        return ContextSnapshot(
            contributions=(ContextContribution(key="active_document", text=self.current),)
            if self.current
            else ()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
async def test_current_document_refreshes_without_rewriting_run_background(
    tmp_path: Path, persistent: bool
) -> None:
    """工具改变宿主状态后下步重采，BCI 保留最初背景，快照不进入历史。"""
    source = _DocumentSource()
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="open-b", name="change_document")),
        tool_response(ToolUseBlock(id="close", name="change_document")),
        text_response("报告 A 检查完成"),
    )
    store = SQLiteStore(tmp_path / "source.db") if persistent else InMemoryLifecycleStore()
    runner = AgentRunner.from_config(
        AgentConfig.model_validate(
            {
                "name": "document-review",
                "model": "openai/test",
                "system": "根据用户原始目标检查报告。",
                "permissions": {"workspace": str(tmp_path)},
            }
        ),
        provider=provider,
        store=store,
        context_source=source,
    )
    runner.runtime.environment.context_input = runner.runtime.environment.context_input.model_copy(
        update={
            "before_current_input": ContextSection(
                slots=[ContextSlot(name="task_background", content="用户针对报告 A 发起检查")]
            )
        }
    )
    changes = 0

    def change_document() -> str:
        """模拟宿主应用切换文档，然后关闭文档。"""
        nonlocal changes
        changes += 1
        source.current = "当前打开报告 B，没有选区" if changes == 1 else ""
        return "宿主状态已更新"

    runner.runtime.environment.tool_bridge.tool_view.registry.register_function(change_document)
    try:
        result = await runner.start(
            AgentRunRequest(input="检查报告 A", run_id="review-a", session_id="documents")
        )
        assert result.run.stop_reason.value == "completed"
        assert result.assistant_message.text == "报告 A 检查完成"
        assert changes == 2
        assert [scope.step_index for scope in source.scopes] == [0, 1, 2]
        assert all(scope.session_id == "documents" for scope in source.scopes)
        assert all(scope.run_id == "review-a" for scope in source.scopes)
        assert all(scope.run_input == "检查报告 A" for scope in source.scopes)
        assert all(scope.workspace_root == tmp_path for scope in source.scopes)

        sidecars = []
        for request in provider.requests:
            snapshots = [
                message
                for message in request.messages
                if message.metadata.get("context_kind") == "runtime_snapshot"
            ]
            assert len(snapshots) == 1
            assert snapshots[0].sender == "context"
            assert request.messages[-1] == snapshots[0]
            assert "用户针对报告 A 发起检查" in "\n".join(
                message.text for message in request.messages
            )
            sidecars.append(snapshots[0].text)
        assert "当前打开报告 A，选中第二段" in sidecars[0]
        assert "当前打开报告 B，没有选区" in sidecars[1]
        assert "当前打开报告 A，选中第二段" not in sidecars[1]
        assert "active_document" not in sidecars[2]
        assert "报告 B" not in sidecars[2]
        assert sidecars[2]

        history = runner.get_session("documents").messages
        assert all(
            message.metadata.get("context_kind") != "runtime_snapshot" for message in history
        )
        history_text = "\n".join(message.text for message in history)
        assert history_text.count("用户针对报告 A 发起检查") == 1
        assert "当前打开报告" not in history_text
        assert result.run.usage.model_steps_committed == 3
        assert result.run.usage.tool_calls_committed == 2
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_one_runner_collects_concurrent_session_snapshots_independently(
    tmp_path: Path,
) -> None:
    """同一 source 可并发采集，不用共享最近状态或全局串行锁。"""
    arrived: list[str] = []
    both_collecting = asyncio.Event()

    class SessionSource:
        """根据调用 scope 返回每个应用会话自己的状态。"""

        async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
            """两份采集同时在场后才返回，明确暴露串行化错误。"""
            arrived.append(scope.session_id)
            if len(arrived) == 2:
                both_collecting.set()
            await both_collecting.wait()
            return ContextSnapshot(
                contributions=(
                    ContextContribution(key="selection", text=f"selection:{scope.session_id}"),
                )
            )

    provider = StaticProvider(text_response(), text_response())
    runner = AgentRunner.from_config(
        AgentConfig.model_validate(
            {
                "name": "concurrent-source",
                "model": "openai/test",
                "system": "检查本会话的选区。",
                "permissions": {"workspace": str(tmp_path)},
            }
        ),
        provider=provider,
        context_source=SessionSource(),
    )
    try:
        async with asyncio.timeout(10):
            results = await asyncio.gather(
                runner.start(AgentRunRequest(input="task:one", session_id="one")),
                runner.start(AgentRunRequest(input="task:two", session_id="two")),
            )
        assert sorted(arrived) == ["one", "two"]
        assert all(result.run.stop_reason.value == "completed" for result in results)
        assert len(provider.requests) == 2
        for request in provider.requests:
            text = "\n".join(message.text for message in request.messages)
            own_session, other_session = ("one", "two") if "task:one" in text else ("two", "one")
            assert f"selection:{own_session}" in text
            assert f"selection:{other_session}" not in text
        for session_id in arrived:
            assert all(
                message.metadata.get("context_kind") != "runtime_snapshot"
                for message in runner.get_session(session_id).messages
            )
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_child_does_not_inherit_parent_source_or_snapshot(tmp_path: Path) -> None:
    """公开路径入口注入父 source，真实 child 只使用自己的任务材料。"""
    source = _DocumentSource()
    parent = StaticProvider(
        tool_response(ToolUseBlock(id="delegate", name="subagent", input={"prompt": "检查标题"})),
        text_response("父任务完成"),
    )
    child = StaticProvider(text_response("标题检查完成"))
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=parent,
        child_provider_factory=ChildProviders(child),
        context_source=source,
    )
    try:
        result = await runner.start(AgentRunRequest(input="检查报告 A", run_id="parent"))
        assert result.run.stop_reason.value == "completed"
        assert len(source.scopes) == 2
        assert all(scope.run_id == "parent" for scope in source.scopes)
        assert all(
            request.messages[-1].metadata.get("context_kind") == "runtime_snapshot"
            for request in parent.requests
        )
        assert len(child.requests) == 1
        assert all(
            message.metadata.get("context_kind") != "runtime_snapshot"
            and source.current not in message.text
            for message in child.requests[0].messages
        )
        assert runner.list_tool_calls("parent")[0].result.model_content == "标题检查完成"
    finally:
        await runner.aclose()
