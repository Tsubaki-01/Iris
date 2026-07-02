"""单轮 Agent runtime。

本模块从 session history、context 和当前用户输入构造一次 `LLMRequest`，
调用注入的 provider，并把 assistant 回复写回 session。
memory 只支持显式 opt-in 注入；工具执行只做一次 bridge，bounded loop 留给后续阶段组合。

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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from ..agents import AgentConfig
from ..context import ContextBuilder, ContextBuildInput
from ..exceptions import IrisError
from ..message import LLMRequest, LLMResponse, Msg
from ..session import InMemorySessionStore, SessionStore
from ..tools import (
    DefaultPermissionPolicy,
    PermissionPolicy,
    ToolExecutor,
    ToolRegistry,
    ToolRegistryView,
    ToolResult,
)
from .assembler import RuntimeMessageAssembler
from .memory import prepare_memory_context_input
from .models import (
    RuntimeErrorInfo,
    RuntimeErrorSource,
    RuntimeOptions,
    RuntimeStatus,
    RuntimeTurnResult,
    ToolBridgeResult,
    ToolErrorPolicy,
)
from .tool_bridge import ToolBridge

# endregion

if TYPE_CHECKING:
    from ..memory import MemoryContextBuilder, MemoryService


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
        tool_registry: ToolRegistry | None = None,
        tool_view: ToolRegistryView | None = None,
        tool_executor: ToolExecutor | None = None,
        workspace_root: Path | None = None,
        permission_policy: PermissionPolicy | None = None,
        memory_service: MemoryService | None = None,
        memory_context_builder: MemoryContextBuilder | None = None,
    ) -> None:
        """创建单轮 runtime。

        Args:
            agent_config (AgentConfig): 已加载并通过配置校验的 Agent 配置。
            context_input (ContextBuildInput): ContextBuilder 的输入数据。
            provider (RuntimeProvider): 本轮调用使用的 provider 实现。
            session_store (SessionStore | None): 可选会话存储；默认使用内存 store。
            context_builder (ContextBuilder | None): 可选 context 构建器，便于测试注入。
            assembler (RuntimeMessageAssembler | None): 可选消息装配器，便于测试注入。
            tool_registry (ToolRegistry | None): 可选工具注册表，供后续工具桥阶段复用。
            tool_view (ToolRegistryView | None): 可选工具视图，默认由注册表创建。
            tool_executor (ToolExecutor | None): 可选工具执行器，默认使用同一注册表和权限策略。
            workspace_root (Path | None): 工具执行时使用的 workspace 根路径。
            permission_policy (PermissionPolicy | None): 工具权限策略。
            memory_service (MemoryService | None): 显式 memory 阶段复用的可选服务。
            memory_context_builder (MemoryContextBuilder | None): 显式 memory 结果裁剪器。
        """
        from ..memory import MemoryContextBuilder

        self.agent_config = agent_config
        self.context_input = context_input
        self.provider = provider
        self.session_store = session_store or InMemorySessionStore()
        self.context_builder = context_builder or ContextBuilder()
        self.assembler = assembler or RuntimeMessageAssembler()
        self.tool_registry = tool_registry or ToolRegistry()
        self.tool_view = tool_view or self.tool_registry.view()
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.permission_policy = permission_policy or DefaultPermissionPolicy()
        self.tool_executor = tool_executor or ToolExecutor(
            self.tool_registry,
            permission_policy=self.permission_policy,
        )
        self.tool_bridge = ToolBridge(
            tool_view=self.tool_view,
            tool_executor=self.tool_executor,
        )
        self.memory_service = memory_service
        self.memory_context_builder = memory_context_builder or MemoryContextBuilder()

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
            context_input = prepare_memory_context_input(
                self.context_input,
                options=runtime_options,
                memory_service=self.memory_service,
                memory_context_builder=self.memory_context_builder,
            )
            context_output = self.context_builder.build(context_input)
            current_input = Msg.user(user_input)
            request = self.assembler.build_request(
                agent_config=self.agent_config,
                context_output=context_output,
                history=history,
                current_input=current_input,
            )
            request = _apply_request_options(request, runtime_options.request_options)
            request = _apply_tool_schemas(
                request,
                include_tools=runtime_options.include_tools,
                tool_view=self.tool_view,
                provider=self.agent_config.model.provider,
            )
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
            bridge_result = await self.tool_bridge.execute_once(
                assistant_message=assistant_message,
                session_id=session_id,
                run_id=run_id,
                step_index=0,
                agent_id=self.agent_config.name,
                workspace_root=self.workspace_root,
                permission_mode=self.agent_config.permissions.writes,
                session_store=self.session_store,
                metadata=run_metadata,
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
            tool_result_messages=bridge_result.messages,
            tool_results=bridge_result.results,
            steps=1,
            metadata=run_metadata,
        )

    async def run_loop(
        self,
        user_input: str,
        *,
        options: RuntimeOptions | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeTurnResult:
        """执行有界 tool loop。

        当前用户输入只进入第一步。后续步骤从 session history 重新装配请求，
        因此上一轮保存的 tool result message 会自然进入下一次 provider 请求。

        Args:
            user_input (str): 当前用户输入内容。
            options (RuntimeOptions | None): 本轮 session、run id、request 和 loop 选项。
            metadata (Mapping[str, Any] | None): 调用方传入的追踪信息。

        Returns:
            RuntimeTurnResult: loop 的最终助手消息、工具结果和状态。
        """
        runtime_options = options or RuntimeOptions()
        run_metadata = {**runtime_options.metadata, **dict(metadata or {})}
        session_id = runtime_options.session_id
        run_id = runtime_options.run_id
        max_steps = runtime_options.loop.max_steps
        all_tool_results: list[ToolResult] = []
        all_tool_messages: list[Msg] = []
        latest_assistant: Msg | None = None
        latest_response: LLMResponse | None = None

        for step_index in range(max_steps):
            step_number = step_index + 1
            current_input = Msg.user(user_input) if step_index == 0 else None
            try:
                history = _load_history(self.session_store, session_id)
                context_input = prepare_memory_context_input(
                    self.context_input,
                    options=runtime_options,
                    memory_service=self.memory_service,
                    memory_context_builder=self.memory_context_builder,
                )
                context_output = self.context_builder.build(context_input)
                request = self.assembler.build_request(
                    agent_config=self.agent_config,
                    context_output=context_output,
                    history=history,
                    current_input=current_input,
                )
                request = _apply_request_options(
                    request,
                    runtime_options.request_options,
                )
                request = _apply_tool_schemas(
                    request,
                    include_tools=runtime_options.include_tools,
                    tool_view=self.tool_view,
                    provider=self.agent_config.model.provider,
                )
                latest_response = await self.provider.complete(request)
                latest_assistant = latest_response.to_msg()
            except Exception as exc:
                return _error_result(
                    session_id=session_id,
                    run_id=run_id,
                    error=normalize_runtime_error(exc),
                    assistant_message=latest_assistant,
                    steps=step_number,
                    metadata=run_metadata,
                )

            messages = [*history]
            if current_input is not None:
                messages.append(current_input)
            messages.append(latest_assistant)

            try:
                self.session_store.save_messages(
                    session_id,
                    [message.model_dump(mode="json") for message in messages],
                )

                if not latest_assistant.has_tool_calls:
                    self.session_store.save_run_metadata(
                        session_id,
                        _build_run_metadata(
                            existing=self.session_store.load_run_metadata(session_id),
                            session_id=session_id,
                            run_id=run_id,
                            status=RuntimeStatus.OK,
                            provider=self.agent_config.model.provider,
                            response=latest_response,
                            message_count=len(messages),
                            metadata=run_metadata,
                            steps=step_number,
                            tool_count=len(all_tool_results),
                        ),
                    )
                    return RuntimeTurnResult(
                        session_id=session_id,
                        run_id=run_id,
                        status=RuntimeStatus.OK,
                        assistant_message=latest_assistant,
                        tool_result_messages=all_tool_messages,
                        tool_results=all_tool_results,
                        steps=step_number,
                        metadata=run_metadata,
                    )

                bridge_result = await self.tool_bridge.execute_once(
                    assistant_message=latest_assistant,
                    session_id=session_id,
                    run_id=run_id,
                    step_index=step_index,
                    agent_id=self.agent_config.name,
                    workspace_root=self.workspace_root,
                    permission_mode=self.agent_config.permissions.writes,
                    session_store=self.session_store,
                    metadata=run_metadata,
                )
                messages.extend(bridge_result.messages)
                self.session_store.save_messages(
                    session_id,
                    [message.model_dump(mode="json") for message in messages],
                )
            except Exception as exc:
                return _error_result(
                    session_id=session_id,
                    run_id=run_id,
                    error=normalize_runtime_error(exc),
                    assistant_message=latest_assistant,
                    steps=step_number,
                    metadata=run_metadata,
                )

            all_tool_results.extend(bridge_result.results)
            all_tool_messages.extend(bridge_result.messages)

            if _should_stop_on_tool_error(runtime_options, bridge_result):
                error = _tool_error_info(bridge_result)
                try:
                    self.session_store.save_run_metadata(
                        session_id,
                        _build_run_metadata(
                            existing=self.session_store.load_run_metadata(session_id),
                            session_id=session_id,
                            run_id=run_id,
                            status=RuntimeStatus.ERROR,
                            provider=self.agent_config.model.provider,
                            response=latest_response,
                            message_count=len(messages),
                            metadata=run_metadata,
                            steps=step_number,
                            tool_count=len(all_tool_results),
                            error=error,
                        ),
                    )
                except Exception as exc:
                    return _error_result(
                        session_id=session_id,
                        run_id=run_id,
                        error=normalize_runtime_error(exc),
                        assistant_message=latest_assistant,
                        steps=step_number,
                        metadata=run_metadata,
                    )
                return RuntimeTurnResult(
                    session_id=session_id,
                    run_id=run_id,
                    status=RuntimeStatus.ERROR,
                    assistant_message=latest_assistant,
                    tool_result_messages=all_tool_messages,
                    tool_results=all_tool_results,
                    steps=step_number,
                    error=error,
                    metadata=run_metadata,
                )

        error = RuntimeErrorInfo(
            code="MAX_STEPS_REACHED",
            message=f"已达到最大 loop 步数: {max_steps}",
            source="runtime",
            details={"max_steps": max_steps},
        )
        max_step_metadata = {**run_metadata, "max_steps": max_steps}
        try:
            self.session_store.save_run_metadata(
                session_id,
                _build_run_metadata(
                    existing=self.session_store.load_run_metadata(session_id),
                    session_id=session_id,
                    run_id=run_id,
                    status=RuntimeStatus.MAX_STEPS,
                    provider=self.agent_config.model.provider,
                    response=latest_response,
                    message_count=len(self.session_store.load_messages(session_id)),
                    metadata=max_step_metadata,
                    steps=max_steps,
                    tool_count=len(all_tool_results),
                    error=error,
                ),
            )
        except Exception as exc:
            return _error_result(
                session_id=session_id,
                run_id=run_id,
                error=normalize_runtime_error(exc),
                assistant_message=latest_assistant,
                steps=max_steps,
                metadata=max_step_metadata,
            )
        return RuntimeTurnResult(
            session_id=session_id,
            run_id=run_id,
            status=RuntimeStatus.MAX_STEPS,
            assistant_message=latest_assistant,
            tool_result_messages=all_tool_messages,
            tool_results=all_tool_results,
            steps=max_steps,
            error=error,
            metadata=max_step_metadata,
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


def _apply_tool_schemas(
    request: LLMRequest,
    *,
    include_tools: bool,
    tool_view: ToolRegistryView,
    provider: str,
) -> LLMRequest:
    """按当前活动工具视图挂载 provider 请求工具 schema。"""
    if not include_tools:
        return request
    api_style = request.provider_options.get("api_style")
    tools = tool_view.active_schemas(
        provider=provider,
        api_style=api_style if isinstance(api_style, str) else None,
    )
    return request.model_copy(update={"tools": tools})


def _build_run_metadata(
    *,
    existing: dict[str, object],
    session_id: str,
    run_id: str,
    status: RuntimeStatus,
    provider: str,
    response: LLMResponse | None,
    message_count: int,
    metadata: Mapping[str, Any],
    steps: int = 1,
    tool_count: int = 0,
    error: RuntimeErrorInfo | None = None,
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
        "model": cast(object, response.model if response is not None else ""),
        "finish_reason": cast(
            object,
            response.finish_reason if response is not None else "",
        ),
        "input_tokens": response.input_tokens if response is not None else 0,
        "output_tokens": response.output_tokens if response is not None else 0,
        "total_tokens": response.total_tokens if response is not None else 0,
        "message_count": message_count,
        "steps": steps,
        "tool_count": tool_count,
        **dict(metadata),
    }
    if error is not None:
        latest_run["error"] = error.model_dump(mode="json")
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
    steps: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> RuntimeTurnResult:
    """构造统一失败结果。"""
    return RuntimeTurnResult(
        session_id=session_id,
        run_id=run_id,
        status=RuntimeStatus.ERROR,
        assistant_message=assistant_message,
        steps=steps,
        error=error,
        metadata=dict(metadata or {}),
    )


def _should_stop_on_tool_error(
    options: RuntimeOptions,
    bridge_result: ToolBridgeResult,
) -> bool:
    """判断 loop 是否应在工具错误后停止。"""
    return options.loop.tool_error_policy == ToolErrorPolicy.STOP and any(
        result.is_error for result in bridge_result.results
    )


def _tool_error_info(bridge_result: ToolBridgeResult) -> RuntimeErrorInfo:
    """从第一个工具错误构造 runtime 错误信息。

    并行执行时不一定是调用顺序上的第一个失败
    """
    for result in bridge_result.results:
        if result.is_error and result.error is not None:
            return RuntimeErrorInfo(
                code=result.error.code,
                message=result.error.message,
                source="tool",
                details=result.error.details,
            )
    return RuntimeErrorInfo(
        code="TOOL_ERROR",
        message="工具执行失败",
        source="tool",
    )


def _classify_runtime_error(error: Exception) -> tuple[str, RuntimeErrorSource]:
    """从 Iris 异常实例读取 runtime 错误映射。"""
    if isinstance(error, IrisError):
        return error.runtime_code, error.runtime_source
    return "RUNTIME_ERROR", "runtime"


__all__ = ["AgentRuntime", "RuntimeProvider", "normalize_runtime_error"]
