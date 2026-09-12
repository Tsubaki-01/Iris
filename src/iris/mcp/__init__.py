"""MCP 接入；导入包不读取凭据、不连接 server。"""

from .config import load_mcp_config
from .manager import MCPManager

__all__ = ["MCPManager", "load_mcp_config"]
