from __future__ import annotations

from typing import Any


class IrisError(Exception):
    """所有 Iris 特定错误的基类。"""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


# ----- 核心 / 配置 / 校验 领域 -----


class IrisConfigError(IrisError, ValueError):
    """配置出现问题时抛出，例如缺少必需的参数或值无效。"""


class IrisValidationError(IrisError):
    """输入或配置校验失败时抛出。"""


class IrisParserError(IrisError):
    """解析结构化内容失败时抛出。"""


class IrisExecutionError(IrisError):
    """任务执行过程中发生异常时抛出。"""


# ----- 提供者 (Provider) 领域 -----


class IrisProviderError(IrisError):
    """模型提供者和 LLM 集成错误的基类。"""


class IrisAPIConnectionError(IrisProviderError):
    """连接到提供者 API 失败时抛出。"""


class IrisRateLimitExceededError(IrisProviderError):
    """超出提供者 API 速率限制时抛出。"""


class IrisAuthenticationError(IrisProviderError):
    """提供者 API 身份认证失败时抛出。"""


# ----- 工具 (Tool) 领域 -----


class IrisToolError(IrisError):
    """工具相关错误的基类。"""


class IrisToolNotFoundError(IrisToolError):
    """请求调用的工具未找到时抛出。"""


class IrisToolExecutionError(IrisToolError):
    """工具执行失败时抛出。"""


class IrisToolValidationError(IrisToolError):
    """工具参数或状态无效时抛出。"""


# ----- MCP 领域 -----


class IrisMCPError(IrisError):
    """MCP 集成错误的基类。"""


class IrisMCPConnectionError(IrisMCPError):
    """连接 MCP 服务器失败时抛出。"""


class IrisMCPProtocolError(IrisMCPError):
    """发生 MCP 协议违规或收到意外响应时抛出。"""


# ----- 代理 (Agent) 领域 -----


class IrisAgentError(IrisError):
    """Agent 相关错误的基类。"""


class IrisAgentExecutionError(IrisAgentError):
    """核心 Agent 执行或主循环失败时抛出。"""


# ----- 记忆 (Memory) 领域 -----


class IrisMemoryError(IrisError):
    """记忆子系统错误的基类。"""


# ----- 模板 (Template) 领域 -----


class IrisTemplateError(IrisError):
    """模板相关错误的基类。"""


class IrisTemplateNotFoundError(IrisTemplateError):
    """找不到所需模板时抛出。"""


class IrisTemplateRenderError(IrisTemplateError):
    """渲染模板期间发生错误时抛出。"""
