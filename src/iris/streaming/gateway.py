"""Exact-session streaming gateway 与 durable sync 组合层。

Gateway 不拥有认证、网络连接或 durable mutation；host 完成授权后，才把 typed command
交给本模块。

Example:
    gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=broker,
        session_id="default",
        durable_page_size=64,
    )
"""

# region imports

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import assert_never

from ..exceptions import IrisError, IrisRunConflictError, IrisRunStateError
from ..harness import AgentRunner, SessionManager
from ..lifecycle import RunResult, RunToolCallRecord
from ..message import ToolResultBlock, ToolUseBlock
from .broker import LiveStreamBroker, LiveSubscription
from .models import (
    CancelAccepted,
    CancelCommand,
    CommandReceipt,
    CommandRejected,
    DurableRunCursor,
    DurableRunPage,
    DurableSync,
    DurableSyncItem,
    GatewayCommand,
    GatewayStreamItem,
    LiveEnvelope,
    LiveSubscriptionRequest,
    ResumeAccepted,
    ResumeCommand,
    SubmitAccepted,
    SubmitCommand,
    SubscribeCommand,
    SyncAccepted,
    SyncCommand,
)

# endregion

logger = logging.getLogger(__name__)


class GatewaySubscription(AsyncIterator[GatewayStreamItem]):
    """先交付可选 durable snapshot，再消费 broker live subscription。"""

    def __init__(
        self,
        *,
        subscription: LiveSubscription,
        stream_epoch: str,
        request: SubscribeCommand,
        initial_sync: DurableSync | None,
        allow_thinking: bool,
        allow_tool_arguments: bool,
    ) -> None:
        """绑定一个 broker subscription 与当前 gateway disclosure policy。"""
        self._subscription = subscription
        self._initial_item = (
            DurableSyncItem(sync=initial_sync) if initial_sync is not None else None
        )
        self._allow_thinking = allow_thinking
        self._allow_tool_arguments = allow_tool_arguments
        self._closed = False
        self.stream_epoch = stream_epoch
        self.scope = request.scope
        self.scope_id = request.scope_id

    def __aiter__(self) -> GatewaySubscription:
        """返回当前 subscription iterator。"""
        return self

    async def __anext__(self) -> GatewayStreamItem:
        """返回下一条 policy-filtered gateway stream item。"""
        if self._closed:
            raise StopAsyncIteration
        if self._initial_item is not None:
            item = self._initial_item
            self._initial_item = None
            return item
        while True:
            try:
                item = await anext(self._subscription)
            except StopAsyncIteration:
                self._closed = True
                raise
            if not isinstance(item, LiveEnvelope):
                return item
            filtered = self._filter_envelope(item)
            if filtered is not None:
                return filtered

    async def aclose(self) -> None:
        """幂等关闭 observation，不触发任何 run cancellation。"""
        if self._closed:
            return
        self._closed = True
        self._initial_item = None
        await self._subscription.aclose()

    def _filter_envelope(self, envelope: LiveEnvelope) -> LiveEnvelope | None:
        """应用 gateway disclosure policy，不重新解析 trusted payload。"""
        channel = envelope.payload.get("channel")
        block_kind = envelope.payload.get("block_kind")
        if not self._allow_thinking and (
            channel == "thinking" or block_kind == "thinking"
        ):
            return None
        if not self._allow_tool_arguments and channel in {
            "tool_name",
            "tool_arguments",
        }:
            return None
        if self._allow_tool_arguments or "tool_name" not in envelope.payload:
            return envelope
        payload = dict(envelope.payload)
        payload.pop("tool_name", None)
        return envelope.model_copy(update={"payload": payload})


class StreamingGateway:
    """绑定 exact runner、manager、broker 与 session 的 typed gateway。"""

    def __init__(
        self,
        *,
        runner: AgentRunner,
        manager: SessionManager,
        broker: LiveStreamBroker,
        session_id: str,
        durable_page_size: int,
        allow_thinking: bool = False,
        allow_tool_arguments: bool = False,
    ) -> None:
        """验证一次 session/capacity binding，不读取 store 或启动 task。

        Args:
            runner (AgentRunner): Exact durable read owner。
            manager (SessionManager): Exact-session command owner。
            broker (LiveStreamBroker): 唯一 live fan-out owner。
            session_id (str): Host 已授权的 bound session。
            durable_page_size (int): 每个 run page 的最大 event 数。
            allow_thinking (bool): 是否向远端暴露 thinking channel。
            allow_tool_arguments (bool): 是否向远端暴露工具参数与 live tool name。

        Raises:
            IrisRunStateError: Session id 空白或 page size 非正整数。
            IrisRunConflictError: Manager 绑定了不同 session。
        """
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            raise IrisRunStateError("session_id 不能为空")
        if manager._session_id != normalized_session_id:
            raise IrisRunConflictError("SessionManager 不属于 gateway bound session")
        if (
            isinstance(durable_page_size, bool)
            or not isinstance(durable_page_size, int)
            or durable_page_size <= 0
        ):
            raise IrisRunStateError("durable_page_size 必须是正整数")
        self._runner = runner
        self._manager = manager
        self._broker = broker
        self._session_id = normalized_session_id
        self._durable_page_size = durable_page_size
        self._allow_thinking = allow_thinking
        self._allow_tool_arguments = allow_tool_arguments

    @property
    def session_id(self) -> str:
        """返回 host 已授权并绑定的 session identity。"""
        return self._session_id

    def subscribe(self, request: SubscribeCommand) -> GatewaySubscription:
        """验证 scope binding，并创建无 network task 的 gateway subscription。

        Args:
            request (SubscribeCommand): 已在 raw boundary 校验的订阅请求。

        Returns:
            GatewaySubscription: 可选先交付 durable sync 的 observation iterator。

        Raises:
            IrisRunConflictError: Scope 或 known run 不属于 bound session。
            IrisRunNotFoundError: 由 runner 在 run 不存在时抛出。
        """
        self._require_scope_binding(request.scope, request.scope_id)
        initial_sync = (
            self.durable_sync(request.durable_cursors)
            if request.durable_cursors
            else None
        )
        subscription = self._broker.subscribe(
            LiveSubscriptionRequest(
                scope=request.scope,
                scope_id=request.scope_id,
                cursor=request.cursor,
            )
        )
        return GatewaySubscription(
            subscription=subscription,
            stream_epoch=self._broker.current_epoch(),
            request=request,
            initial_sync=initial_sync,
            allow_thinking=self._allow_thinking,
            allow_tool_arguments=self._allow_tool_arguments,
        )

    async def handle(self, command: GatewayCommand) -> CommandReceipt:
        """把 typed command 路由到 exact manager 或 durable read facade。"""
        try:
            if isinstance(command, SubmitCommand):
                receipt = await self._manager.submit(
                    command.input,
                    mode=command.mode,
                    options=command.options,
                )
                return SubmitAccepted(request_id=command.request_id, receipt=receipt)
            if isinstance(command, ResumeCommand):
                result = await self._manager.resume(
                    interaction_id=command.interaction_id,
                    response=command.response,
                )
                return ResumeAccepted(request_id=command.request_id, result=result)
            if isinstance(command, CancelCommand):
                run = await self._manager.interrupt(reason=command.reason)
                return CancelAccepted(request_id=command.request_id, run=run)
            if isinstance(command, SyncCommand):
                return SyncAccepted(
                    request_id=command.request_id,
                    sync=self.durable_sync(command.cursors),
                )
            assert_never(command)
        except IrisError as exc:
            return CommandRejected(
                request_id=command.request_id,
                command_kind=command.kind,
                code=exc.runtime_code,
                message=exc.message,
            )
        except Exception as exc:
            logger.warning(
                "streaming gateway command 处理失败",
                extra={
                    "request_id": command.request_id,
                    "command_kind": command.kind,
                    "session_id": self._session_id,
                    "exception_type": type(exc).__qualname__,
                },
            )
            return CommandRejected(
                request_id=command.request_id,
                command_kind=command.kind,
                code="INTERNAL_ERROR",
                message="命令处理失败",
            )

    def durable_sync(self, cursors: Sequence[DurableRunCursor]) -> DurableSync:
        """按 caller-known run 顺序读取有限 durable pages。"""
        self._runner.get_session(self._session_id)
        authorized = []
        for cursor in cursors:
            run = self._runner.get_run(cursor.run_id)
            if run.session_id != self._session_id:
                raise IrisRunConflictError("run 不属于 gateway bound session")
            authorized.append((cursor, run))

        pages: list[DurableRunPage] = []
        for cursor, run in authorized:
            result = self._runner.get_result(run.run_id)
            tool_calls = self._runner.list_tool_calls(run.run_id)
            events = self._runner.list_events(
                run.run_id,
                cursor.after_sequence,
                limit=self._durable_page_size + 1,
            )
            page_events = tuple(events[: self._durable_page_size])
            next_cursor = (
                DurableRunCursor(
                    run_id=run.run_id,
                    after_sequence=page_events[-1].sequence,
                )
                if len(events) > self._durable_page_size
                else None
            )
            pages.append(
                DurableRunPage(
                    run=run,
                    result=self._filter_result(result),
                    tool_calls=tuple(self._filter_tool_calls(tool_calls)),
                    events=page_events,
                    next_cursor=next_cursor,
                )
            )
        return DurableSync(session_id=self._session_id, runs=tuple(pages))

    def _require_scope_binding(self, scope: str, scope_id: str) -> None:
        """验证 subscription scope 只属于 bound session。"""
        if scope == "session":
            if scope_id != self._session_id:
                raise IrisRunConflictError("session scope 不属于 gateway bound session")
            return
        run = self._runner.get_run(scope_id)
        if run.session_id != self._session_id:
            raise IrisRunConflictError("run scope 不属于 gateway bound session")

    def _filter_tool_calls(
        self,
        tool_calls: Sequence[RunToolCallRecord],
    ) -> list[RunToolCallRecord]:
        """投影 durable tool arguments 与 remote-safe result fields。"""
        filtered: list[RunToolCallRecord] = []
        for call in tool_calls:
            updates: dict[str, object] = {}
            if not self._allow_tool_arguments:
                updates["arguments"] = {}
            if call.result is not None:
                result = call.result
                error = (
                    result.error.model_copy(update={"details": {}})
                    if result.error is not None
                    else None
                )
                updates["result"] = result.model_copy(
                    update={
                        "error": error,
                        "data": {},
                        "artifact": None,
                        "stats": {},
                        "metadata": {},
                    }
                )
            filtered.append(call.model_copy(update=updates) if updates else call)
        return filtered

    def _filter_result(self, result: RunResult | None) -> RunResult | None:
        """投影 remote-safe result，并按 policy 处理工具参数。"""
        if result is None:
            return result
        updates: dict[str, object] = {}
        assistant = result.assistant_message
        if assistant is not None:
            content = assistant.content
            if not isinstance(content, str):
                content = [
                    block.model_copy(update={"input": {}})
                    if isinstance(block, ToolUseBlock) and not self._allow_tool_arguments
                    else block.model_copy(update={"metadata": {}})
                    if isinstance(block, ToolResultBlock)
                    else block
                    for block in content
                ]
            updates["assistant_message"] = assistant.model_copy(
                update={"content": content, "metadata": {}}
            )
        interaction = result.pending_interaction
        if interaction is not None:
            tool_call_updates: dict[str, object] = {"workspace_root": "<redacted>"}
            if not self._allow_tool_arguments:
                tool_call_updates["arguments"] = {}
            tool_call = interaction.request.tool_call.model_copy(update=tool_call_updates)
            request = interaction.request.model_copy(update={"tool_call": tool_call})
            updates["pending_interaction"] = interaction.model_copy(
                update={"request": request}
            )
        if result.error is not None:
            updates["error"] = result.error.model_copy(update={"details": {}})
        return result.model_copy(update=updates) if updates else result


__all__ = ["GatewaySubscription", "StreamingGateway"]
