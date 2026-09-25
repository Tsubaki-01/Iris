"""SDK 已取得确定结果后的本地投影与 artifact 收口。"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from mcp import types

from iris.exceptions import IrisToolExecutionError
from iris.mcp.models import MCPResolvedServer
from iris.tools import ToolArtifact, ToolExecutionContext, ToolResult
from iris.tools.artifacts import ToolArtifactStore

from .fixtures.tools import make_tool


@pytest.mark.asyncio
@pytest.mark.parametrize("write_fails", [False, True])
async def test_mcp_projection_drains_after_sdk_result_without_recalling_server(
    stdio_config: MCPResolvedServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_fails: bool,
) -> None:
    """完整投影在线程执行，取消和落盘失败均不重放已返回的远端调用。"""
    tool, connection = make_tool(stdio_config)
    connection.result = types.CallToolResult(
        content=[types.TextContent(type="text", text="known")],
        structured_content={"answer": "value"},
    )
    context = ToolExecutionContext(workspace_root=tmp_path, call_id="mcp-projection")
    entered = threading.Event()
    release = threading.Event()
    workers: list[int] = []
    loop_thread = threading.get_ident()
    project = tool._project_result

    def blocked_projection(
        result: types.CallToolResult, context: ToolExecutionContext
    ) -> ToolResult:
        workers.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        return project(result, context)

    def fail_persist(self: ToolArtifactStore, *args: object, **kwargs: object) -> ToolArtifact:
        raise IrisToolExecutionError("ARTIFACT_ERROR: controlled write failure")

    monkeypatch.setattr(tool, "_project_result", blocked_projection)
    if write_fails:
        monkeypatch.setattr(ToolArtifactStore, "_persist_text", fail_persist)
    execution = asyncio.create_task(tool.arun({}, context))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        assert workers[0] != loop_thread
        execution.cancel()
        await asyncio.sleep(0)
        execution.cancel()
        await asyncio.sleep(0)
        assert not execution.done()
    finally:
        release.set()
        [result] = await asyncio.gather(execution, return_exceptions=True)
    assert not isinstance(result, BaseException)
    assert len(connection.calls) == 1
    if write_fails:
        assert result.error.code == "ARTIFACT_ERROR"
    else:
        assert result.artifact is not None
        assert '"answer": "value"' in result.artifact.path.read_text(encoding="utf-8")
