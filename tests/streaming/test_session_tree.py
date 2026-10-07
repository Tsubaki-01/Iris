"""会话树首包、原身份与持久补读的确定性集成。"""

from pathlib import Path

import pytest

from iris.harness import AgentRunner, AgentRunRequest, SessionManager
from iris.message import ToolUseBlock
from iris.streaming import LiveStreamBroker, StreamingGateway
from iris.streaming.models import LiveEnvelope, SubscribeCommand

from ..harness.fakes import text_response, tool_response
from ..harness.test_runner_subagent import (
    ChildProviders,
    StreamingStaticProvider,
    _write_configs,
)


@pytest.mark.asyncio
async def test_tree_subscription_sees_short_child_from_first_fact(tmp_path: Path) -> None:
    """先订阅 root tree，再执行极短 child，不等待父卡片出现才订阅 child。"""
    broker = LiveStreamBroker(replay_capacity_per_scope=256, subscription_capacity=256)
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path),
        provider=StreamingStaticProvider(
            tool_response(ToolUseBlock(id="delegate", name="subagent", input={"prompt": "child"})),
            text_response("parent"),
        ),
        live_publisher=broker,
        child_provider_factory=ChildProviders(StreamingStaticProvider(text_response("child"))),
    )
    manager = SessionManager(runner, "root", submission_publisher=broker)
    gateway = StreamingGateway(
        runner=runner, manager=manager, broker=broker, session_id="root", durable_page_size=50
    )
    tree = gateway.subscribe(
        SubscribeCommand(request_id="tree", scope="session_tree", scope_id="root")
    )
    exact = gateway.subscribe(
        SubscribeCommand(request_id="exact", scope="session", scope_id="root")
    )
    await runner.start(AgentRunRequest(input="delegate", session_id="root", run_id="parent"))
    child = runner.list_child_runs("parent").items[0]
    tree_items = []
    async for item in tree:
        tree_items.append(item)
        if (
            isinstance(item, LiveEnvelope)
            and item.kind == "run.terminal"
            and item.run_id == "parent"
        ):
            break
    child_items = [item for item in tree_items if item.run_id == child.run.run_id]
    assert child_items[0].kind == "subagent.linked"
    assert any(item.kind == "model.response.started" for item in child_items)
    assert all(item.session_id == child.run.session_id for item in child_items)
    assert all(item.lineage.parent_run_id == "parent" for item in child_items)
    async for item in exact:
        assert item.run_id == "parent"
        if item.kind == "run.terminal":
            break
    snapshot = gateway.durable_snapshot([child.run.run_id])
    assert snapshot.runs[0].run.run_id == child.run.run_id
    await tree.aclose()
    await exact.aclose()
    await manager.close()
    await runner.aclose()
