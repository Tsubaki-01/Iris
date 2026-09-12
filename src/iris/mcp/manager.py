"""一套 runtime 的 MCP 准备、固定目录发布与资源关闭。"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from ..exceptions import IrisConfigError, IrisMCPError
from ..tools.registry import ToolRegistry
from .catalog import build_catalog
from .config import resolve_server_config
from .connection import MCPConnection
from .models import MCPCatalogSnapshot, MCPConfig, MCPDiagnostic, MCPServerSnapshot
from .tools import MCPTool

logger = logging.getLogger(__name__)


async def _close_connection(connection: MCPConnection) -> Exception | None:
    """记录单连接关闭失败，允许 owner 继续清理其余连接。"""
    try:
        await connection.aclose()
    except Exception as error:
        logger.warning("MCP server %s 关闭失败: %s", connection.config.server_id, error)
        return error
    return None


class MCPManager:
    """准备/关闭与 snapshot 的唯一 owner；不拥有 run 或工具 effect。"""

    def __init__(self, config: MCPConfig, *, registry: ToolRegistry, workspace_root: Path) -> None:
        self.config = config
        self.registry = registry
        self.workspace_root = workspace_root
        self._lock = asyncio.Lock()
        self._closed = False
        self._connections: list[MCPConnection] = []
        self._snapshot: MCPCatalogSnapshot | None = None

    @property
    def snapshot(self) -> MCPCatalogSnapshot | None:
        """返回唯一固定快照；准备前为 None。"""
        return self._snapshot

    async def prepare(self) -> MCPCatalogSnapshot:
        """串行准备完整候选，原子发布后返回固定目录。

        Raises:
            IrisConfigError: required server 环境解析失败。
            IrisMCPError: required server 连接/发现失败，或 manager 已关闭。
        """
        async with self._lock:
            if self._closed:
                raise IrisMCPError("MCP manager 已关闭，请创建新的 runtime")
            if self._snapshot is not None:
                return self._snapshot
            try:
                self._snapshot = await self._prepare_catalog()
            except BaseException:
                self._closed = True
                await self._close_owned()
                raise
            return self._snapshot

    async def _prepare_catalog(self) -> MCPCatalogSnapshot:
        """所有候选仅驻留局部变量，registry 是名称 admission 的唯一边界。"""
        environ = dict(os.environ)
        diagnostics = list(self.config.diagnostics)
        servers: list[MCPServerSnapshot] = []
        candidates: list[MCPTool] = []
        loop = asyncio.get_running_loop()
        for server in sorted(self.config.servers, key=lambda item: item.server_id):
            connection: MCPConnection | None = None
            deadline = loop.time() + server.startup_timeout_sec
            try:
                async with asyncio.timeout_at(deadline):
                    config = resolve_server_config(
                        server, environ=environ, workspace_root=self.workspace_root
                    )
                    connection = MCPConnection(config)
                    self._connections.append(connection)
                    await connection.open()
                    sdk_tools = await connection.list_tools()
                    tools, tool_diagnostics = build_catalog(config, sdk_tools)
                    # 同步 resolver/schema 编译不会让出 loop，发布前计入同一个 startup 期限。
                    if loop.time() >= deadline:
                        raise TimeoutError
            except (IrisConfigError, IrisMCPError, TimeoutError) as error:
                failure = (
                    IrisMCPError("MCP server 准备超时", server_id=server.server_id)
                    if isinstance(error, TimeoutError)
                    else error
                )
                if connection is not None:
                    await _close_connection(connection)
                    self._connections.pop()
                if server.required:
                    raise failure from None
                stage = "config" if isinstance(failure, IrisConfigError) else "connect"
                diagnostics.append(
                    MCPDiagnostic(
                        server.server_id,
                        stage,
                        failure.runtime_code,
                        str(failure),
                        failure.context.get("field"),
                    )
                )
                logger.warning("跳过 optional MCP server %s: %s", server.server_id, failure)
                continue
            diagnostics.extend(tool_diagnostics)
            servers.append(MCPServerSnapshot(config, connection.protocol_version, tools))
            candidates.extend(MCPTool(tool, connection) for tool in tools)
        self.registry.register_many(candidates)
        return MCPCatalogSnapshot(tuple(servers), tuple(diagnostics))

    async def _close_owned(self) -> Exception | None:
        """每个连接尝试关闭一次；返回首个错误且不跳过剩余资源。"""
        connections, self._connections = self._connections, []
        first_error: Exception | None = None
        for connection in reversed(connections):
            error = await _close_connection(connection)
            if first_error is None:
                first_error = error
        return first_error

    async def aclose(self) -> None:
        """等待准备收尾后关闭全部资源；显式关闭失败报告给 host。"""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            error = await self._close_owned()
            if error is not None:
                raise IrisMCPError("MCP 资源关闭失败") from error


__all__ = ["MCPManager"]
