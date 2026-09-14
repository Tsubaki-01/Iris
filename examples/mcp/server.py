"""无需凭据的本地 STDIO MCP 示例，由 Iris 启动并关闭。"""

from datetime import UTC, datetime, timedelta, timezone
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("iris-local-example")


@server.tool(annotations=ToolAnnotations(read_only_hint=True), structured_output=False)
def echo(text: str) -> str:
    """原样返回文本。

    Args:
        text: 要回传的文本。

    Returns:
        与输入相同的文本。
    """
    return text


@server.tool(annotations=ToolAnnotations(read_only_hint=True), structured_output=True)
def get_current_time(
    timezone_name: Literal["Asia/Shanghai", "UTC"] = "Asia/Shanghai",
) -> dict[str, str | int]:
    """读取本机时钟，返回北京时间或 UTC，精确到秒。

    Args:
        timezone_name: Asia/Shanghai 表示北京时间，UTC 表示协调世界时。

    Returns:
        包含 timezone、带偏移量的 iso_time 和 unix_timestamp 的结构化结果。
    """
    zone = timezone(timedelta(hours=8)) if timezone_name == "Asia/Shanghai" else UTC
    current = datetime.now(zone).replace(microsecond=0)
    return {
        "timezone": timezone_name,
        "iso_time": current.isoformat(),
        "unix_timestamp": int(current.timestamp()),
    }


if __name__ == "__main__":
    server.run(transport="stdio")
