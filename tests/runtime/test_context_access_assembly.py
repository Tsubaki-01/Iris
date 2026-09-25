"""配置装配与工具结果原文回读的正常运行链。"""

from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.exceptions import IrisConfigError
from iris.harness import AgentRunner, AgentRunRequest
from iris.harness._context_access import ContextAccess
from iris.harness.session_history import SessionHistory
from iris.message import LLMResponse, TextBlock, ToolUseBlock
from iris.runtime import RuntimeFactory
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools.context_access import ContextReadInput
from tests.harness.fakes import StaticProvider


def _config(workspace: Path, *, enabled: bool = True) -> AgentConfig:
    """构造只使用当前阶段策略的 agent。"""
    return AgentConfig.model_validate(
        {
            "name": "reader",
            "model": "openai/test",
            "system": "读取材料",
            "permissions": {"workspace": str(workspace)},
            "context_policy": {"enabled": enabled},
        }
    )


def test_low_level_factory_requires_explicit_access(tmp_path: Path) -> None:
    """默认启用的低层工厂不能假装持有 lifecycle store。"""
    config = _config(tmp_path)
    with pytest.raises(IrisConfigError, match="context_access"):
        RuntimeFactory.from_config(config, provider=StaticProvider())
    runtime = RuntimeFactory.from_config(
        config, provider=StaticProvider(), context_access=ContextAccess(InMemoryLifecycleStore())
    )
    assert {tool.name for tool in runtime.environment.tool_bridge.tool_view.active_tools} == {
        "context_read",
        "context_search",
    }
    disabled = RuntimeFactory.from_config(
        _config(tmp_path, enabled=False), provider=StaticProvider()
    )
    assert disabled.environment.tool_bridge.tool_view.active_tools == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_runner_reads_offload_without_file_tool_and_fork_keeps_ref(
    tmp_path: Path,
    backend: str,
) -> None:
    """真实 executor/store/model loop 完成产物回读，fork 不复制或重跑原工具。"""
    provider = StaticProvider(
        LLMResponse(provider="test", content=[ToolUseBlock(id="produce", name="produce")]),
        LLMResponse(
            provider="test",
            content=[
                ToolUseBlock(
                    id="read", name="context_read", input={"ref": "result:2:0", "limit": 8000}
                )
            ],
        ),
        LLMResponse(provider="test", content=[TextBlock(text="done")]),
    )
    store = (
        SQLiteStore(tmp_path / "history.db") if backend == "sqlite" else InMemoryLifecycleStore()
    )
    runner = AgentRunner.from_config(_config(tmp_path), provider=provider, store=store)
    calls = 0
    full_text = "原始材料\r\n" * 700

    def produce() -> str:
        """返回一次较大的普通业务结果。"""
        nonlocal calls
        calls += 1
        return full_text

    registry = runner.runtime.environment.tool_bridge.tool_view.registry
    tool = registry.register_function(produce)
    tool.definition.max_result_chars = 500
    try:
        result = await runner.start(AgentRunRequest(input="读取大材料", session_id="main"))
        assert result.run.stop_reason.value == "completed"
        assert calls == 1
        assert "read_file" not in {tool.name for tool in registry._active_tools}
        original = store.read_session_messages("main", start=2, limit=1).items[0][1]
        assert "历史原文：result:2:0" not in original.tool_results[0].content
        assert "历史原文：result:2:0" in provider.requests[1].messages[3].tool_results[0].content
        read_result = runner.list_tool_calls(result.run.run_id)[1].result
        assert read_result is not None and read_result.data["content"] == full_text
        assert read_result.artifact is None
        branch = SessionHistory(store).fork(result.run.run_id)
        restored_store = SQLiteStore(tmp_path / "history.db") if backend == "sqlite" else store
        page = ContextAccess(restored_store).read(
            branch.session_id, ContextReadInput(ref="result:2:0", limit=8000), tmp_path
        )
        assert page.content == full_text
        assert calls == 1
    finally:
        await runner.aclose()
