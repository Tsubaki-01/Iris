"""通过真实会话、暂停与恢复验证工具披露不改变共享注册表。"""

from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.context import ContextBuildScope, ContextSnapshot
from iris.harness import AgentRunner, AgentRunRequest
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import ForkSession, LifecycleStore
from iris.message import LLMRequest, ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .fakes import FrozenClock, StaticProvider, text_response, tool_batch_response, tool_response


def _schema_names(request: LLMRequest) -> set[str]:
    """读取真正交给 provider 的完整 schema 名称。"""
    return {schema["function"]["name"] for schema in request.tools}


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
async def test_search_disclosure_survives_pause_without_leaking_between_sessions(
    tmp_path: Path, persistent: bool
) -> None:
    """另一会话搜索不改 waiting 批次，重开 SQLite 后按已保存可见集合执行。"""
    provider = StaticProvider(
        tool_response(
            ToolUseBlock(
                id="discover-alpha", name="tool_search", input={"query": "alpha", "limit": 1}
            )
        ),
        tool_batch_response(
            ToolUseBlock(id="ask", name="ask_question", input={"question": "继续检查？"}),
            ToolUseBlock(id="use-alpha", name="alpha", input={"document": "A"}),
        ),
        tool_response(
            ToolUseBlock(
                id="discover-beta", name="tool_search", input={"query": "beta", "limit": 1}
            )
        ),
        text_response("第二会话完成"),
        text_response("第一会话完成"),
    )
    collected: list[tuple[str, int]] = []
    executed: list[str] = []

    class Source:
        """记录主步骤采集，恢复工具批次本身不能再采集。"""

        async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
            """返回空的当前状态，并记录精确步骤归属。"""
            collected.append((scope.session_id, scope.step_index))
            return ContextSnapshot()

    def alpha(document: str) -> str:
        """检查 alpha 文档的完整内容。"""
        executed.append(document)
        return f"检查完成:{document}"

    def beta(document: str) -> str:
        """检查 beta 文档的完整内容。"""
        executed.append(f"unexpected:{document}")
        return document

    config = AgentConfig.model_validate(
        {
            "name": "deferred-documents",
            "model": "openai/test",
            "system": "先发现工具，再按完整 schema 调用。",
            "permissions": {"workspace": str(tmp_path)},
            "tools": {"builtin": ["human.ask"]},
            "context_policy": {"deferred_tools": True},
        }
    )

    def make_runner(store: LifecycleStore) -> AgentRunner:
        """每次宿主启动注册同一份工具定义，不恢复任何进程内披露缓存。"""
        runner = AgentRunner.from_config(
            config, provider=provider, store=store, context_source=Source()
        )
        registry = runner.runtime.environment.tool_bridge.tool_view.registry
        registry.register_function(alpha, deferred=True)
        registry.register_function(beta, deferred=True)
        return runner

    store = SQLiteStore(tmp_path / "deferred.db") if persistent else InMemoryLifecycleStore()
    runner = make_runner(store)
    try:
        waiting = await runner.start(
            AgentRunRequest(input="检查 A", session_id="one", run_id="first")
        )
        assert waiting.pending_interaction is not None
        assert not executed
        initial_names, discovered_names = map(_schema_names, provider.requests)
        assert "tool_search" in initial_names
        assert not {"alpha", "beta"} & initial_names
        assert "alpha" in discovered_names and "beta" not in discovered_names
        schema = next(
            item["function"]
            for item in provider.requests[1].tools
            if item["function"]["name"] == "alpha"
        )
        assert schema["parameters"]["properties"]["document"]["type"] == "string"
        assert schema["parameters"]["required"] == ["document"]
        checkpoint = store.load_checkpoint("first")
        assert set(checkpoint.engine_cursor["visible_tool_names"]) == discovered_names

        second = await runner.start(
            AgentRunRequest(input="发现 beta", session_id="two", run_id="second")
        )
        assert second.run.stop_reason.value == "completed"
        assert not {"alpha", "beta"} & _schema_names(provider.requests[2])
        assert "beta" in _schema_names(provider.requests[3])
        assert "alpha" not in _schema_names(provider.requests[3])
        assert store.load_checkpoint("first") == checkpoint
        assert not runner.runtime.environment.tool_bridge.tool_view.allow

        if persistent:
            await runner.aclose()
            store = SQLiteStore(tmp_path / "deferred.db")
            runner = make_runner(store)
        result = await runner.resume(
            "first",
            interaction_id=waiting.pending_interaction.interaction_id,
            response=QuestionInteractionResponse(answer="继续"),
        )
        assert result.run.stop_reason.value == "completed"
        assert result.assistant_message.text == "第一会话完成"
        assert executed == ["A"]
        assert "alpha" in _schema_names(provider.requests[4])
        assert "beta" not in _schema_names(provider.requests[4])
        assert collected == [("one", 0), ("one", 1), ("two", 0), ("two", 1), ("one", 2)]
        assert not runner.runtime.environment.tool_bridge.tool_view.allow
        records = runner.list_tool_calls("first")
        assert len(records) == 3
        assert records[-1].result.model_content == "检查完成:A"
    finally:
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("pressure", [False, True])
async def test_fork_rebuilds_only_copied_discovery_and_final_consumes_protection(
    tmp_path: Path, pressure: bool
) -> None:
    """终态前缀只带已发生的发现，已完成搜索回合不永久保护首项 schema。"""

    class PressureProvider(StaticProvider):
        """先正常完成发现，再仅在分支请求中制造可选 schema 压力。"""

        pressure = False

        def estimate_input_tokens(self, request: LLMRequest) -> int:
            """保留整包成本，alpha 的完整 schema 使请求越过压力线。"""
            base = 7400 if self.pressure else 1
            return (
                base
                + len(request.model_dump_json()) // 100
                + (1000 if self.pressure and "alpha" in _schema_names(request) else 0)
            )

    provider = PressureProvider(
        tool_response(
            ToolUseBlock(id="find-a", name="tool_search", input={"query": "alpha", "limit": 1})
        ),
        text_response("已发现 alpha"),
        tool_response(
            ToolUseBlock(id="find-b", name="tool_search", input={"query": "beta", "limit": 1})
        ),
        text_response("已发现 beta"),
        text_response("分支完成"),
    )
    clock = FrozenClock()
    runner = AgentRunner.from_config(
        AgentConfig.model_validate(
            {
                "name": "fork-discovery",
                "model": "openai/test",
                "system": "按需发现工具。",
                "permissions": {"workspace": str(tmp_path)},
                "context_policy": {"deferred_tools": True},
                "compaction": {"input_budget_tokens": 10000},
            }
        ),
        provider=provider,
        clock=clock,
    )

    def lookup() -> str:
        """返回工具内容。"""
        return "已查询"

    registry = runner.runtime.environment.tool_bridge.tool_view.registry
    registry.register_function(lookup, name="alpha", deferred=True)
    registry.register_function(lookup, name="beta", deferred=True)
    try:
        for name in ("alpha", "beta"):
            result = await runner.start(
                AgentRunRequest(input=f"查找 {name}", run_id=name, session_id="main")
            )
            assert result.run.stop_reason.value == "completed"
        fork = runner.store.fork_session(
            ForkSession(source_run_id="alpha", target_session_id="branch", now=clock.now())
        )
        assert len(fork.messages) < len(runner.get_session("main").messages)
        provider.pressure = pressure
        branch = await runner.start(AgentRunRequest(input="继续", session_id="branch"))
        assert branch.run.stop_reason.value == "completed"
        assert len(provider.requests) == 5
        names = _schema_names(provider.requests[-1])
        assert "beta" not in names
        assert ("alpha" in names) is not pressure
        assert branch.run.usage.compaction.total_tokens == 0
    finally:
        await runner.aclose()
