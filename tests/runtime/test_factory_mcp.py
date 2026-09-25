"""共享装配只读取声明，环境与外部连接在 prepare 时处理。"""

from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.runtime import RuntimeFactory

from ..harness.fakes import StaticProvider
from ..mcp.fixtures.runtime import MCPPeer


@pytest.mark.asyncio
@pytest.mark.parametrize("from_path", [False, True])
async def test_factory_uses_config_base_and_prepares_existing_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    from_path: bool,
) -> None:
    peer = MCPPeer(monkeypatch)
    (tmp_path / "mcp.json").write_text('{"servers":{"test":{"command":"${IRIS_FIXTURE_CMD}"}}}')
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: a\nmodel: openai/test\nsystem: test\n"
        "context_policy:\n  enabled: false\nmcp:\n  path: mcp.json\n"
    )
    if from_path:
        runtime = RuntimeFactory.from_config_path(path, provider=StaticProvider())
    else:
        config = AgentConfig.model_validate(
            {
                "name": "a",
                "model": "openai/test",
                "system": "test",
                "context_policy": {"enabled": False},
                "mcp": {"path": "mcp.json"},
            }
        )
        runtime = RuntimeFactory.from_config(config, config_path=path, provider=StaticProvider())
    view = runtime.environment.tool_bridge.tool_view
    assert not peer.events and not view.active_tools
    monkeypatch.setenv("IRIS_FIXTURE_CMD", "fixture")
    snapshot = await runtime.environment.aprepare()
    assert snapshot.servers[0].config.command == "fixture"
    assert [tool.name for tool in view.active_tools] == ["mcp__test__echo"]
    await runtime.environment.aclose()
    assert peer.events == ["open", "list", "close"]
