"""实际启动双格式 MCP 示例；LLM 使用离线 provider。"""

import json
from pathlib import Path

import pytest

from iris.agents import load_agent_config
from iris.exceptions import IrisMCPError
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest, RunStopReason, ToolCallPhase
from iris.message import ToolUseBlock

from ..harness.fakes import StaticProvider, text_response, tool_response

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "mcp"


@pytest.mark.asyncio
async def test_json_and_toml_examples_execute_same_local_tool(tmp_path: Path) -> None:
    for filename in ("mcp.json", "mcp.toml"):
        config = load_agent_config(EXAMPLE / "agent.yaml")
        config = config.model_copy(
            update={
                "mcp": config.mcp.model_copy(update={"path": EXAMPLE / filename}),
                "permissions": config.permissions.model_copy(update={"workspace": str(tmp_path)}),
            }
        )
        runner = AgentRunner.from_config(
            config,
            config_path=EXAMPLE / "agent.yaml",
            provider=StaticProvider(
                tool_response(
                    ToolUseBlock(id="echo", name="mcp__local__echo", input={"text": "你好 MCP"})
                ),
                text_response("done"),
            ),
        )
        try:
            result = await runner.start(AgentRunRequest(input="echo", run_id=filename))
            assert result.run.stop_reason is RunStopReason.COMPLETED
            record = runner.list_tool_calls(filename)[0]
            assert record.phase is ToolCallPhase.COMMITTED
            assert record.result.model_content.startswith("你好 MCP")
            artifact = json.loads(record.result.artifact.path.read_text(encoding="utf-8"))
            assert artifact["content"][0]["text"] == "你好 MCP"
        finally:
            await runner.aclose()


@pytest.mark.asyncio
async def test_missing_example_runtime_reports_required_server_without_run(tmp_path: Path) -> None:
    config = load_agent_config(EXAMPLE / "agent.yaml")
    declaration = json.loads((EXAMPLE / "mcp.json").read_text())
    declaration["mcpServers"]["local"]["command"] = "iris-example-python-does-not-exist"
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(declaration))
    config = config.model_copy(update={"mcp": config.mcp.model_copy(update={"path": path})})
    runner = AgentRunner.from_config(config, provider=StaticProvider())
    with pytest.raises(IrisMCPError) as caught:
        await runner.start(AgentRunRequest(input="echo", run_id="missing"))
    assert caught.value.context["server_id"] == "local"
    assert runner.store.load_run("missing") is None
