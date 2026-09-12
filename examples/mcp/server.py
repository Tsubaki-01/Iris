"""无需凭据的本地 STDIO MCP 示例，由 Iris 启动并关闭。"""

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


if __name__ == "__main__":
    server.run(transport="stdio")
