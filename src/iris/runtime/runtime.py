"""单轮 Agent runtime。

本模块从 session history、context 和当前用户输入构造一次 `LLMRequest`，
调用注入的 provider，并把 assistant 回复写回 session。
这里刻意不接入 memory、工具执行或 bounded loop，避免单轮链路承担后续阶段职责。

Example:
    runtime = AgentRuntime(
        agent_config=config,
        context_input=context_input,
        provider=fake_provider,
    )
    result = await runtime.run_turn("你好")
"""

# region imports
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

from ..agents import AgentConfig
from ..context import ContextBuilder, ContextBuildInput
from ..exceptions import IrisError
from ..message import LLMRequest, LLMResponse, Msg
from ..session import InMemorySessionStore, SessionStore
from .assembler import RuntimeMessageAssembler
from .models import (
    RuntimeErrorInfo,
    RuntimeErrorSource,
    RuntimeOptions,
    RuntimeStatus,
    RuntimeTurnResult,
)

# endregion


class RuntimeProvider(Protocol):
    """Runtime 调用的 provider 最小协议。

    运行时只依赖 provider-neutral 的 `LLMRequest` / `LLMResponse`，这样测试可注入
    FakeProvider，生产路径也可复用真实 provider client，而不让 runtime 读取厂商
    raw payload。

    Example:
        response = await provider.complete(request)
    """

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """执行一次非流式 LLM 请求。

        Args:
            request (LLMRequest): Runtime 已组装完成的 provider-neutral 请求。

        Returns:
            LLMResponse: Provider 适配层归一化后的响应。
        """


class AgentRuntime:
    """编排一次本地 Agent 单轮调用。

    `AgentRuntime` 只负责调用顺序和边界错误归一化；context 构建、消息装配、
    provider 调用和 session 存储仍由各自组件承担。这样单轮 runtime 可先稳定下来，
    后续 memory、工具桥和 loop 可以在同一边界上继续组合。

    Attributes:
        agent_config (AgentConfig): 已校验的 Agent 配置。
        context_input (ContextBuildInput): 本轮使用的 context 输入定义。
        provider (RuntimeProvider): 注入的 provider 实现。
        session_store (SessionStore): 会话历史和 run metadata 存储。
        context_builder (ContextBuilder): Context 构建器。
        assembler (RuntimeMessageAssembler): Provider 请求装配器。

    Example:
        runtime = AgentRuntime(
            agent_config=config,
            context_input=context_input,
            provider=fake_provider,
        )
        result = await runtime.run_turn("当前问题")
    """

    def __init__(
        self,
        *,
        agent_config: AgentConfig,
        context_input: ContextBuildInput,
        provider: RuntimeProvider,
        session_store: SessionStore | None = None,
        context_builder: ContextBuilder | None = None,
        assembler: RuntimeMessageAssembler | None = None,
    ) -> None:
        """创建单轮 runtime。

        Args:
            agent_config (AgentConfig): 已加载并通过配置校验的 Agent 配置。
            context_input (ContextBuildInput): ContextBuilder 的输入数据。
            provider (RuntimeProvider): 本轮调用使用的 provider 实现。
            session_store (SessionStore | None): 可选会话存储；默认使用内存 store。
            context_builder (ContextBuilder | None): 可选 context 构建器，便于测试注入。
            assembler (RuntimeMessageAssembler | None): 可选消息装配器，便于测试注入。
        """
        self.agent_config = agent_config
        self.context_input = context_input
        self.provider = provider
        self.session_store = session_store or InMemorySessionStore()
        self.context_builder = context_builder or ContextBuilder()
        self.assembler = assembler or RuntimeMessageAssembler()

    async def run_turn(
        self,
        user_input: str,
        *,
        options: RuntimeOptions | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeTurnResult:
        """执行一次 provider call 并保存用户输入与 assistant 回复。

        Args:
            user_input (str): 当前用户输入内容。
            options (RuntimeOptions | None): 本轮 session、run id 和 request 覆盖选项。
            metadata (Mapping[str, Any] | None): 调用方传入的追踪信息，只进入 run metadata。

        Returns:
            RuntimeTurnResult: 成功时包含 assistant 消息；失败时包含归一化错误。
        """
        runtime_options = options or RuntimeOptions()
        run_metadata = {**runtime_options.metadata, **dict(metadata or {})}
        session_id = runtime_options.session_id
        run_id = runtime_options.run_id

        # --- 1. Build request ---
        try:
            history = _load_history(self.session_store, session_id)
            context_output = self.context_builder.build(self.context_input)
            current_input = Msg.user(user_input)
            request = self.assembler.build_request(
                agent_config=self.agent_config,
                context_output=context_output,
                history=history,
                current_input=current_input,
            )
            request = _apply_request_options(request, runtime_options.request_options)
        except Exception as exc:
            return _error_result(
                session_id=session_id,
                run_id=run_id,
                error=normalize_runtime_error(exc),
                metadata=run_metadata,
            )

        # --- 2. Call provider ---
        try:
            response = await self.provider.complete(request)
            assistant_message = response.to_msg()
        except Exception as exc:
            return _error_result(
                session_id=session_id,
                run_id=run_id,
                error=normalize_runtime_error(exc),
                metadata=run_metadata,
            )

        # --- 3. Persist result ---
        messages = [*history, current_input, assistant_message]
        try:
            self.session_store.save_messages(
                session_id,
                [message.model_dump(mode="json") for message in messages],
            )
            self.session_store.save_run_metadata(
                session_id,
                _build_run_metadata(
                    existing=self.session_store.load_run_metadata(session_id),
                    session_id=session_id,
                    run_id=run_id,
                    status=RuntimeStatus.OK,
                    provider=self.agent_config.model.provider,
                    response=response,
                    message_count=len(messages),
                    metadata=run_metadata,
                ),
            )
        except Exception as exc:
            return _error_result(
                session_id=session_id,
                run_id=run_id,
                error=normalize_runtime_error(exc),
                assistant_message=assistant_message,
                metadata=run_metadata,
            )

        return RuntimeTurnResult(
            session_id=session_id,
            run_id=run_id,
            status=RuntimeStatus.OK,
            assistant_message=assistant_message,
            steps=1,
            metadata=run_metadata,
        )


def normalize_runtime_error(error: Exception) -> RuntimeErrorInfo:
    """将 runtime 边界异常归一化为稳定错误信息。

    Args:
        error (Exception): Runtime 边界捕获到的异常。

    Returns:
        RuntimeErrorInfo: 可放入 `RuntimeTurnResult.error` 的结构化错误。
    """
    code, source = _classify_runtime_error(error)
    details: dict[str, Any] = {}
    if isinstance(error, IrisError):
        details.update(error.context)
    return RuntimeErrorInfo(
        code=code,
        message=str(error),
        source=source,
        details=details,
    )


def _load_history(session_store: SessionStore, session_id: str) -> list[Msg]:
    """从 session 读取历史消息并恢复为 `Msg`。"""
    return [Msg.from_dict(message) for message in session_store.load_messages(session_id)]


def _apply_request_options(
    request: LLMRequest,
    request_options: Mapping[str, Any],
) -> LLMRequest:
    """应用本轮请求覆盖项。

    `provider_options` 使用浅合并，保留 Agent model 配置里的 provider 选项，同时允许
    `RuntimeOptions.request_options` 覆盖或补充单次调用字段。
    """
    if not request_options:
        return request
    update = dict(request_options)
    if "provider_options" in update:
        provider_options = update["provider_options"]
        if isinstance(provider_options, Mapping):
            update["provider_options"] = {
                **request.provider_options,
                **dict(provider_options),
            }
    return request.model_copy(update=update)


def _build_run_metadata(
    *,
    existing: dict[str, object],
    session_id: str,
    run_id: str,
    status: RuntimeStatus,
    provider: str,
    response: LLMResponse,
    message_count: int,
    metadata: Mapping[str, Any],
) -> dict[str, object]:
    """构建 session 中保存的 run metadata。

    `latest_run` 便于快速读取最近一次结果，`runs` 保留 append-like 历史，避免 Stage 03
    引入新的持久化表结构或 session model。
    """
    latest_run: dict[str, object] = {
        "session_id": session_id,
        "run_id": run_id,
        "status": status.value,
        "provider": cast(object, provider),
        "model": cast(object, response.model),
        "finish_reason": cast(object, response.finish_reason),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
        "message_count": message_count,
        "steps": 1,  # 目前单轮 runtime 只算一步，后续可以继续修改。
        **dict(metadata),
    }
    runs = existing.get("runs", [])
    run_list = list(runs) if isinstance(runs, list) else []
    run_list.append(latest_run)
    return {**existing, "latest_run": latest_run, "runs": run_list}


def _error_result(
    *,
    session_id: str,
    run_id: str,
    error: RuntimeErrorInfo,
    assistant_message: Msg | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RuntimeTurnResult:
    """构造统一失败结果。"""
    return RuntimeTurnResult(
        session_id=session_id,
        run_id=run_id,
        status=RuntimeStatus.ERROR,
        assistant_message=assistant_message,
        steps=1,
        error=error,
        metadata=dict(metadata or {}),
    )


def _classify_runtime_error(error: Exception) -> tuple[str, RuntimeErrorSource]:
    """从 Iris 异常实例读取 runtime 错误映射。"""
    if isinstance(error, IrisError):
        return error.runtime_code, error.runtime_source
    return "RUNTIME_ERROR", "runtime"


__all__ = ["AgentRuntime", "RuntimeProvider", "normalize_runtime_error"]
