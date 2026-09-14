"""通过真实 STDIO MCP 验证本地时间工具，模型使用确定性 provider。"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris.agents import load_agent_config
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest, RunStopReason, ToolCallPhase
from iris.message import ToolUseBlock

from ..harness.fakes import StaticProvider, text_response, tool_response

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "mcp"


def time_runner(tmp_path: Path, arguments: dict[str, str]) -> AgentRunner:
    """复用示例配置，把工具结果写入本次测试目录。"""
    config = load_agent_config(EXAMPLE / "agent.yaml")
    config = config.model_copy(
        update={"permissions": config.permissions.model_copy(update={"workspace": str(tmp_path)})}
    )
    return AgentRunner.from_config(
        config,
        config_path=EXAMPLE / "agent.yaml",
        provider=StaticProvider(
            tool_response(
                ToolUseBlock(id="time", name="mcp__local__get_current_time", input=arguments)
            ),
            text_response("done"),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments,zone,offset",
    [
        ({}, "Asia/Shanghai", timedelta(hours=8)),
        ({"timezone_name": "Asia/Shanghai"}, "Asia/Shanghai", timedelta(hours=8)),
        ({"timezone_name": "UTC"}, "UTC", timedelta()),
    ],
    ids=["default-beijing", "explicit-beijing", "utc"],
)
async def test_current_time_is_real_and_preserves_timezone(
    tmp_path: Path, arguments: dict[str, str], zone: str, offset: timedelta
) -> None:
    """结构化时间落在真实调用窗口内，ISO 时间与 Unix 时间戳表示同一时刻。"""
    runner = time_runner(tmp_path, arguments)
    try:
        await runner.aprepare()
        snapshot = runner.runtime.environment.mcp_manager.snapshot
        assert any(tool.wire_name == "get_current_time" for tool in snapshot.servers[0].tools)
        before = int(datetime.now(UTC).timestamp())
        result = await runner.start(AgentRunRequest(input="查询时间", run_id="time"))
        after = int(datetime.now(UTC).timestamp())
        assert result.run.stop_reason is RunStopReason.COMPLETED
        record = runner.list_tool_calls("time")[0]
        assert record.phase is ToolCallPhase.COMMITTED
        assert record.result.error is None
        artifact = json.loads(record.result.artifact.path.read_text(encoding="utf-8"))
        payload = artifact["structuredContent"]
        moment = datetime.fromisoformat(payload["iso_time"])
        assert payload["timezone"] == zone
        assert moment.utcoffset() == offset
        assert int(moment.timestamp()) == payload["unix_timestamp"]
        assert before <= payload["unix_timestamp"] <= after
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_unsupported_timezone_returns_tool_error(tmp_path: Path) -> None:
    """未声明的时区由工具 schema 拒绝，不产生伪造时间。"""
    runner = time_runner(tmp_path, {"timezone_name": "Mars/Olympus"})
    try:
        await runner.aprepare()
        snapshot = runner.runtime.environment.mcp_manager.snapshot
        assert any(tool.wire_name == "get_current_time" for tool in snapshot.servers[0].tools)
        result = await runner.start(AgentRunRequest(input="查询时间", run_id="invalid-time"))
        assert result.run.stop_reason is RunStopReason.COMPLETED
        record = runner.list_tool_calls("invalid-time")[0]
        assert record.phase is ToolCallPhase.COMMITTED
        assert record.result.is_error
        assert record.result.error is not None
        assert "Mars/Olympus" in record.result.error.message
    finally:
        await runner.aclose()
