"""MCP 资源、调用与工具协议异常。"""

from typing import Any

from .tools import IrisToolError


class IrisMCPError(IrisToolError):
    """MCP 准备、发现或资源关闭失败。"""

    runtime_error_code = "MCP_ERROR"


class IrisMCPToolError(IrisMCPError):
    """已知 MCP 工具错误，由 adapter 投影为普通工具结果。"""

    def __init__(self, message: str, *, code: str, **context: Any) -> None:
        super().__init__(message, **context)
        self.code = code

    @property
    def runtime_code(self) -> str:
        """返回本次已知工具错误的稳定分类。"""
        return self.code


class IrisMCPCallError(IrisMCPError):
    """SDK 未提供可确认的调用结果，不推断请求是否已发送。"""

    runtime_error_code = "MCP_CALL_FAILED"
