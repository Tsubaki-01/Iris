"""Runner 集成测试的受控外部 MCP 连接，不替换 manager 或内核。"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from mcp import types

from iris.agents import AgentConfig
from iris.exceptions import IrisMCPError
from iris.mcp.models import MCPResolvedServer


class MCPPeer:
    """记录连接、发现和调用，并允许测试控制准备与请求清理时间。"""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.events: list[str] = []
        self.opened = asyncio.Event()
        self.release_open = asyncio.Event()
        self.release_open.set()
        self.called = asyncio.Event()
        self.cleaning = asyncio.Event()
        self.release_cleanup = asyncio.Event()
        self.release_cleanup.set()
        self.block_call = False
        self.return_on_cancel = False
        self.fail_open = False
        self.protocol_version = "2026-07-28"
        self.tools = (
            types.Tool(
                name="echo",
                input_schema={"type": "object"},
                annotations=types.ToolAnnotations(read_only_hint=True),
            ),
        )
        peer = self

        class Connection:
            def __init__(self, config: MCPResolvedServer) -> None:
                self.config = config
                self.protocol_version = peer.protocol_version

            async def open(self) -> None:
                peer.events.append("open")
                peer.opened.set()
                await peer.release_open.wait()
                if peer.fail_open:
                    raise IrisMCPError("fixture unavailable")

            async def list_tools(self) -> tuple[types.Tool, ...]:
                peer.events.append("list")
                return peer.tools

            async def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
                peer.events.append(f"call:{name}")
                peer.called.set()
                if peer.block_call:
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        peer.cleaning.set()
                        await peer.release_cleanup.wait()
                        peer.events.append("cleaned")
                        if not peer.return_on_cancel:
                            raise
                return types.CallToolResult(content=[types.TextContent(type="text", text="echo")])

            async def aclose(self) -> None:
                peer.events.append("close")

        monkeypatch.setattr("iris.mcp.manager.MCPConnection", Connection)


def mcp_agent(tmp_path: Path, *, trust: bool = True) -> AgentConfig:
    """生成合成配置；声明的环境变量到首次 prepare 才求值。"""
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"test": {"command": "fixture"}}}))
    return AgentConfig.model_validate(
        {
            "name": "mcp-agent",
            "model": "openai/fake-model",
            "system": "test",
            "permissions": {"workspace": str(tmp_path)},
            "mcp": {"path": str(path), "overrides": {"test": {"trust_annotations": trust}}},
        }
    )
