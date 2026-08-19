from __future__ import annotations

from typing import Any, ClassVar


class IrisError(Exception):
    """所有 Iris 特定错误的基类。"""

    runtime_error_source: ClassVar[str] = "runtime"
    runtime_error_code: ClassVar[str] = "RUNTIME_ERROR"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    @property
    def runtime_source(self) -> str:
        """返回 runtime 错误来源。"""
        return self.runtime_error_source

    @property
    def runtime_code(self) -> str:
        """返回 runtime 稳定错误码。"""
        return self.runtime_error_code

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


# ----- 核心 / 配置 / 校验 领域 -----


class IrisConfigError(IrisError, ValueError):
    """配置出现问题时抛出，例如缺少必需的参数或值无效。"""

    runtime_error_source = "config"
    runtime_error_code = "CONFIG_ERROR"


class IrisValidationError(IrisError):
    """输入或配置校验失败时抛出。"""


class IrisParserError(IrisError):
    """解析结构化内容失败时抛出。"""


class IrisExecutionError(IrisError):
    """任务执行过程中发生异常时抛出。"""


# ----- Runtime 控制流领域 -----


class IrisCancellationRequestedError(IrisError):
    """Activation 已请求协作式取消时使用的内部控制流异常。"""

    runtime_error_code = "CANCELLATION_REQUESTED"


# ----- Lifecycle / Run 领域 -----


class IrisRunError(IrisError):
    """Logical run 生命周期领域错误的基类。"""

    runtime_error_source = "lifecycle"
    runtime_error_code = "RUN_ERROR"


class IrisRunNotFoundError(IrisRunError):
    """目标 logical run 或其关联记录不存在。"""

    runtime_error_code = "RUN_NOT_FOUND"


class IrisRunConflictError(IrisRunError):
    """Lifecycle identity、lane 或 compare-and-set 发生冲突。"""

    runtime_error_code = "RUN_CONFLICT"


class IrisRunStateError(IrisRunError):
    """Lifecycle command 不适用于当前 run 状态。"""

    runtime_error_code = "RUN_STATE_ERROR"


class IrisRunPersistenceError(IrisRunError):
    """Lifecycle store 无法可靠读取或提交事务。"""

    runtime_error_source = "persistence"
    runtime_error_code = "RUN_PERSISTENCE_ERROR"


class IrisRunRecoveryError(IrisRunError):
    """Durable recovery facts 无法安全解释。"""

    runtime_error_code = "RUN_RECOVERY_ERROR"


class IrisRunObservationTimeoutError(IrisRunError):
    """等待 logical run 可观察状态变化时超时。"""

    runtime_error_code = "RUN_OBSERVATION_TIMEOUT"


class IrisLifecycleSchemaError(IrisRunPersistenceError):
    """Lifecycle 持久化 schema 与当前契约不兼容。"""

    runtime_error_code = "LIFECYCLE_SCHEMA_ERROR"


# ----- Human-in-the-loop 领域 -----


class IrisHITLError(IrisError):
    """人工交互生命周期和恢复协议错误的基类。"""

    runtime_error_source = "runtime"
    runtime_error_code = "HITL_ERROR"


class HITLResponseMismatchError(IrisHITLError):
    runtime_error_code = "HITL_RESPONSE_MISMATCH"


class HITLConflictError(IrisHITLError):
    runtime_error_code = "HITL_CONFLICT"


class HITLCheckpointInvalidError(IrisHITLError):
    runtime_error_code = "HITL_CHECKPOINT_INVALID"


# ----- Context 领域 -----


class IrisContextError(IrisError, ValueError):
    """Context System 相关错误的基类。"""

    runtime_error_source = "context"
    runtime_error_code = "CONTEXT_ERROR"


# ----- Skill 领域 -----


class IrisSkillError(IrisError, ValueError):
    """Skill 子系统错误的基类。"""


class IrisSkillFormatError(IrisSkillError):
    """Skill 文件格式无效。"""


class IrisSkillPathError(IrisSkillError):
    """Skill 路径不满足 workspace 边界。"""


class IrisSkillNotFoundError(IrisSkillError):
    """按名称找不到 Skill。"""


# ----- 提供者 (Provider) 领域 -----


class IrisProviderError(IrisError):
    """模型提供者和 LLM 集成错误的基类。"""

    runtime_error_source = "provider"
    runtime_error_code = "PROVIDER_ERROR"


class IrisAPIConnectionError(IrisProviderError):
    """连接到提供者 API 失败时抛出。"""


class IrisRateLimitExceededError(IrisProviderError):
    """超出提供者 API 速率限制时抛出。"""


class IrisAuthenticationError(IrisProviderError):
    """提供者 API 身份认证失败时抛出。"""


# ----- 工具 (Tool) 领域 -----


class IrisToolError(IrisError):
    """工具相关错误的基类。"""

    runtime_error_source = "tool"
    runtime_error_code = "PROTOCOL_ERROR"


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

    runtime_error_source = "memory"
    runtime_error_code = "MEMORY_ERROR"


# ----- 模板 (Template) 领域 -----


class IrisTemplateError(IrisError):
    """模板相关错误的基类。"""


class IrisTemplateNotFoundError(IrisTemplateError):
    """找不到所需模板时抛出。"""


class IrisTemplateRenderError(IrisTemplateError):
    """渲染模板期间发生错误时抛出。"""
