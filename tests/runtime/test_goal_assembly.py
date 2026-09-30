"""Goal 装配尊重开关、执行范围与现有宿主上下文。"""

from pathlib import Path

import pytest

from iris.agents import AgentConfig, ToolsConfig
from iris.context import ContextBuildScope, ContextContribution, ContextSnapshot
from iris.exceptions import IrisConfigError
from iris.goal import GoalService
from iris.harness import AgentRunner, AgentRunRequest
from iris.harness._context_access import ContextAccess
from iris.message import ToolUseBlock
from iris.runtime import RuntimeFactory
from iris.runtime._assembly import assemble_runtime, resolve_runtime_boundary
from iris.runtime.environment import RuntimeExecutionScope
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolRegistry
from tests.harness.fakes import StaticProvider, text_response, tool_response


class HostSource:
    """提供一次原宿主采集及其选材元数据。"""

    def __init__(self) -> None:
        self.calls = 0
        self.snapshot = ContextSnapshot(
            contributions=(ContextContribution("host", "原宿主状态", required=False, priority=9),)
        )

    async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
        """记录一次真正的宿主采集。"""
        self.calls += 1
        return self.snapshot


def _config(tmp_path: Path, *, enabled: bool) -> AgentConfig:
    """创建当前阶段的配置，保留 context policy 默认值。"""
    return AgentConfig.model_validate(
        {
            "name": "goal-agent",
            "model": "openai/test",
            "system": "完成用户目标",
            "permissions": {"workspace": str(tmp_path)},
            "goal": {"enabled": enabled},
            "context_policy": {"deferred_tools": True},
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "scope"),
    [
        (False, RuntimeExecutionScope.ROOT),
        (True, RuntimeExecutionScope.ROOT),
        (False, RuntimeExecutionScope.CHILD),
    ],
)
async def test_goal_switch_controls_service_tools_and_source(
    tmp_path: Path, enabled: bool, scope: RuntimeExecutionScope
) -> None:
    """显式注入不能绕过关闭开关；开启时组合宿主贡献且仅采集一次。"""
    config = _config(tmp_path, enabled=enabled)
    store = InMemoryLifecycleStore()
    service = GoalService(store, config=config.goal)
    source = HostSource()
    runtime = assemble_runtime(
        config,
        config_path=None,
        provider=StaticProvider(),
        memory_service=None,
        api_key=None,
        execution_scope=scope,
        boundary=resolve_runtime_boundary(config),
        context_access=ContextAccess(store),
        context_source=source,
        goal_service=service,
    )
    environment = runtime.environment
    assert environment.goal_service is (service if enabled else None)
    tools = {tool.name: tool for tool in environment.tool_bridge.tool_view.active_tools}
    assert ("get_goal" in tools) is enabled
    assert ("report_goal" in tools) is enabled
    if enabled:
        for name in ("get_goal", "report_goal"):
            assert not tools[name].definition.deferred
            assert tools[name].definition.context_retention == "keep"
        assert environment.context_source is not source
    else:
        assert environment.context_source is source
    snapshot = await environment.context_source.collect(
        ContextBuildScope("session", "ordinary", 0, tmp_path, "用户输入")
    )
    assert source.calls == 1
    assert snapshot.contributions[0] is source.snapshot.contributions[0]
    assert [item.key for item in snapshot.contributions] == (
        ["host", "iris.goal"] if enabled else ["host"]
    )


@pytest.mark.parametrize("scope", [RuntimeExecutionScope.ROOT, RuntimeExecutionScope.CHILD])
def test_enabled_goal_rejects_missing_owner_or_child(
    tmp_path: Path, scope: RuntimeExecutionScope
) -> None:
    """装配边界负责 service 与 scope，不让 inner engine 猜测 owner。"""
    config = _config(tmp_path, enabled=True)
    store = InMemoryLifecycleStore()
    with pytest.raises(IrisConfigError, match="Goal|goal"):
        assemble_runtime(
            config,
            config_path=None,
            provider=StaticProvider(),
            memory_service=None,
            api_key=None,
            execution_scope=scope,
            boundary=resolve_runtime_boundary(config),
            context_access=ContextAccess(store),
            goal_service=GoalService(store) if scope is RuntimeExecutionScope.CHILD else None,
        )


def test_independent_factory_points_goal_to_runner(tmp_path: Path) -> None:
    """独立工厂没有生命周期存储，不能构造 Goal owner。"""
    with pytest.raises(IrisConfigError, match="AgentRunner"):
        RuntimeFactory.from_config(
            _config(tmp_path, enabled=True),
            provider=StaticProvider(),
            context_access=ContextAccess(InMemoryLifecycleStore()),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("child_enabled", [False, True])
async def test_root_owns_exact_store_and_child_does_not_inherit_goal(
    tmp_path: Path, backend: str, child_enabled: bool
) -> None:
    """真实父子装配复用存储但隔离 Goal，非法 child 沿现有工具错误路径返回。"""
    child_path = tmp_path / "child.yaml"
    child_path.write_text(
        "name: child\nmodel: openai/test\nsystem: child\n"
        + ("goal:\n  enabled: true\n" if child_enabled else ""),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "default: child\nagents:\n  child:\n    path: child.yaml\n    description: child\n",
        encoding="utf-8",
    )
    config = _config(tmp_path, enabled=True).model_copy(
        update={"tools": ToolsConfig(subagent=catalog)}
    )
    store = SQLiteStore(tmp_path / "goal.db") if backend == "sqlite" else InMemoryLifecycleStore()
    child_provider = StaticProvider(text_response())
    parent = AgentRunner.from_config(
        config,
        config_path=tmp_path / "parent.yaml",
        provider=StaticProvider(
            tool_response(ToolUseBlock(id="delegate", name="subagent", input={"prompt": "任务"})),
            text_response(),
        ),
        store=store,
        child_provider_factory=lambda config, *, config_path: child_provider,
    )
    try:
        service = parent.runtime.environment.goal_service
        assert service is not None and service.store is store
        if not child_enabled:
            controller = parent._subagent_controller
            assert controller is not None
            child = controller._assemble_child(controller.routes.routes["child"])
            try:
                environment = child.runtime.environment
                assert environment.goal_service is None
                assert environment.context_source is None
                assert not {"get_goal", "report_goal"}.intersection(
                    tool.name for tool in environment.tool_bridge.tool_view.active_tools
                )
            finally:
                await child.aclose()
        result = await parent.start(AgentRunRequest(input="委派任务", session_id="parent"))
        call = parent.list_tool_calls(result.run.run_id)[0]
        assert call.result is not None
        if child_enabled:
            assert call.result.error.code == "SUBAGENT_CONFIG_ERROR"
            assert not child_provider.requests
        else:
            assert call.result.error is None
            assert len(child_provider.requests) == 1
    finally:
        await parent.aclose()


@pytest.mark.parametrize("name", ["get_goal", "report_goal"])
def test_goal_tool_conflict_is_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """不覆盖用户已有工具，冲突由统一 registry 入口判定。"""
    import iris.runtime._assembly as assembly

    registry = ToolRegistry()

    def custom() -> str:
        """返回用户自己的结果。"""
        return "custom"

    original = registry.register_function(custom, name=name)
    monkeypatch.setattr(assembly, "build_tool_registry", lambda *args, **kwargs: registry)
    config = _config(tmp_path, enabled=True)
    store = InMemoryLifecycleStore()
    with pytest.raises(IrisConfigError, match="get_goal/report_goal"):
        assemble_runtime(
            config,
            config_path=None,
            provider=StaticProvider(),
            memory_service=None,
            api_key=None,
            execution_scope=RuntimeExecutionScope.ROOT,
            boundary=resolve_runtime_boundary(config),
            context_access=ContextAccess(store),
            goal_service=GoalService(store),
        )
    assert registry.get(name) is original
