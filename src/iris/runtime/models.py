"""Runtime 专属数据模型。

本模块只定义 runtime 阶段共享的配置、快照和结果模型，不执行 provider、
工具、memory 或 session 逻辑。
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..context import ContextBuildOutput
from ..memory import MemoryQuery, MemorySearchResult
from ..message import Conversation, LLMRequest, Msg
from ..tools import ToolResult

RuntimeErrorSource = Literal[
    "config",
    "context",
    "provider",
    "tool",
    "memory",
    "session",
    "runtime",
]


def _new_run_id() -> str:
    """生成一次 runtime run 的本地唯一 ID。"""
    return f"run_{uuid.uuid4().hex[:12]}"


class RuntimeStatus(StrEnum):
    """Runtime 单轮或 loop 的结束状态。"""

    OK = "ok"
    ERROR = "error"
    MAX_STEPS = "max_steps"


class ToolErrorPolicy(StrEnum):
    """Loop 遇到工具错误时的处理策略。"""

    RETURN_TO_MODEL = "return_to_model"
    STOP = "stop"


class BoundedLoopOptions(BaseModel):
    """有界 loop 的基础控制参数。"""

    max_steps: int = Field(default=4, gt=0)
    tool_error_policy: ToolErrorPolicy = ToolErrorPolicy.RETURN_TO_MODEL

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class RuntimeErrorInfo(BaseModel):
    """Runtime 对外返回的结构化错误信息。"""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    source: RuntimeErrorSource
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class RuntimeTurnInput(BaseModel):
    """单次 runtime 调用的用户输入包。"""

    user_input: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ToolBridgeResult(BaseModel):
    """一次工具桥接阶段的结果快照。

    `ToolBridge` 会同时生成面向程序、模型和 session 的三份视图，调用方按使用场景
    选择对应字段，而不需要重新转换工具结果。

    Attributes:
        results (list[ToolResult]): 程序侧可读取的结构化工具执行结果。
        messages (list[Msg]): 可回灌给模型的 tool result 消息。
        events (list[dict[str, Any]]): 已写入 session store 的 JSON-safe 工具事件快照。
    """

    results: list[ToolResult] = Field(default_factory=list)
    messages: list[Msg] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class Runstate(BaseModel):
    """Runtime 内部测试和调试用的单步运行快照。

    Attributes:
        session_id (str): 当前运行绑定的会话标识，用于关联 session history。
        run_id (str): 当前 runtime run 的唯一标识，用于串联日志、事件和调试信息。
        step_index (int): 当前 loop 步骤序号，从 0 开始。
        context_output (ContextBuildOutput): 本步请求使用的 context 构建结果。
        history (list[Msg]): 从 session 层读取的历史消息，不包含本轮 context 注入。
        current_input (Msg | None): 本步新增输入；loop 后续步骤可为空。
        conversation (Conversation): assembler 生成的完整 provider 请求消息序列。
        tools_schema (list[dict[str, Any]]): 本步挂载到请求上的工具 schema 快照。
        request (LLMRequest): 本步发送给 provider 的 provider-neutral 请求。
        metadata (dict[str, Any]): 仅用于调试和追踪的运行态附加信息。
    """

    session_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    context_output: ContextBuildOutput
    history: list[Msg] = Field(default_factory=list)
    current_input: Msg | None = None
    conversation: Conversation
    tools_schema: list[dict[str, Any]] = Field(default_factory=list)
    request: LLMRequest
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class RuntimeOptions(BaseModel):
    """Runtime 调用级选项。

    Attributes:
        session_id (str): 本次调用使用的会话标识，默认使用 `"default"`。
        run_id (str): 本次调用的唯一运行标识，默认自动生成。
        include_tools (bool): 是否在 provider 请求中包含可用工具 schema。
        request_options (dict[str, Any]): 透传给 `LLMRequest` 的请求级覆盖项， provider 专属选项。
        metadata (dict[str, Any]): 运行态追踪信息，不直接进入 prompt。
        memory_query (MemoryQuery | None): 显式触发 memory recall 的查询条件。
        memory_results (list[MemorySearchResult] | None): 调用方预先提供的 memory 结果。
        memory_max_chars (int): memory 注入 context 时允许使用的字符预算。
        loop (BoundedLoopOptions): 有界 loop 的步数和工具错误处理配置。
    """

    session_id: str = "default"
    run_id: str = Field(default_factory=_new_run_id)
    include_tools: bool = True
    request_options: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    memory_query: MemoryQuery | None = None
    memory_results: list[MemorySearchResult] | None = None
    memory_max_chars: int = Field(default=4000, gt=0)
    loop: BoundedLoopOptions = Field(default_factory=BoundedLoopOptions)

    model_config = ConfigDict(extra="forbid")


class RuntimeTurnResult(BaseModel):
    """Runtime 对外返回的单轮或 loop 执行结果。

    Attributes:
        session_id (str): 本次结果所属的会话标识。
        run_id (str): 本次结果所属的 runtime run 标识。
        status (RuntimeStatus): 本次运行的最终状态。
        assistant_message (Msg | None): 最终对外返回的 assistant 消息。
        tool_result_messages (list[Msg]): 工具执行后可回灌给模型的消息。
        tool_results (list[ToolResult]): 程序侧可读取的结构化工具执行结果。
        steps (int): 本次运行实际完成的 provider 调用步数。
        error (RuntimeErrorInfo | None): 失败时返回的归一化错误信息。
        metadata (dict[str, Any]): 运行摘要、追踪字段或调试附加信息。
    """

    session_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    status: RuntimeStatus
    assistant_message: Msg | None = None
    tool_result_messages: list[Msg] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    steps: int = Field(default=1, gt=0)
    error: RuntimeErrorInfo | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


__all__ = [
    "BoundedLoopOptions",
    "Runstate",
    "RuntimeErrorInfo",
    "RuntimeOptions",
    "RuntimeStatus",
    "RuntimeTurnInput",
    "RuntimeTurnResult",
    "ToolBridgeResult",
    "ToolErrorPolicy",
]
