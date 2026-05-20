"""工具执行入口。

控制工具逻辑的核心调用执行器模块，负责分发单次和并发批次执行，
捕获异常将其归一化为标准化返回结构，以及拦截并交由策略验证权限。

Example:
    executor = ToolExecutor(registry)
    results = await executor.execute_many(blocks, context)
"""

# region imports
from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from ..exceptions import (
    IrisToolExecutionError,
    IrisToolNotFoundError,
    IrisToolValidationError,
)
from ..message import ToolUseBlock
from .artifacts import ToolArtifactStore
from .base import BaseTool, ToolErrorInfo, ToolExecutionContext, ToolResult
from .circuit import CircuitBreaker
from .permissions import DefaultPermissionPolicy, PermissionPolicy
from .registry import ToolRegistry

# endregion


class ToolExecutor:
    """执行 `ToolUseBlock` 的工具执行器。

    充当模型产生的工具调用原语向实际实现的桥梁，具备参数校验、
    上下文透传、结果加工等防腐层功能。自动将所有不稳定的底层异常进行标准化拦截。

    Attributes:
        registry (ToolRegistry): 工具注册表，用以按名拾取工具。
        permission_policy (PermissionPolicy): 用于执行前权限风控检测卡控。
        artifact_preview_chars (int): 输出结果持久化到硬盘时的预览摘要字数限制。

    Example:
        executor = ToolExecutor(registry)
        res = await executor.execute_one(tool_use_block, ctx)
    """

    # ==========================================
    #               Initialization
    # ==========================================
    # region
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        permission_policy: PermissionPolicy | None = None,
        artifact_preview_chars: int = 8000,
        middleware: Sequence[object] | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """初始化执行器。

        配置注册表、策略及长输出阈值等核心工作设置。

        Args:
            registry (ToolRegistry): 配置好可用方法的当前状态总库。
            permission_policy (PermissionPolicy | None): 安全及交互式授权拦截规则处理器。
            artifact_preview_chars (int): 若被启用硬盘持久化，保留前置内容的字符数。
            middleware (Sequence[object] | None): 工具调用生命周期钩子。
            circuit_breaker (CircuitBreaker | None): 连续失败熔断器。
        """
        self.registry = registry
        self.permission_policy = permission_policy or DefaultPermissionPolicy()
        self.artifact_preview_chars = artifact_preview_chars
        self.middleware = list(middleware or [])
        self.circuit_breaker = circuit_breaker

    # endregion

    # ==========================================
    #               Execute Methods
    # ==========================================
    # region
    async def execute_one(
        self,
        tool_use: ToolUseBlock,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行单个工具调用并将错误归一化为 `ToolResult`。

        提供对调用全周期（取用、拦截、加工、执行与存盘）的集成支撑，
        将一切错误包装在不打断模型运行的专用响应结构中返回。

        Args:
            tool_use (ToolUseBlock): 携带名字、唯一标记与输入参数的模型指令结构。
            context (ToolExecutionContext): 生命周期环境，提供跨工具所需的文件及会话根环境。

        Returns:
            ToolResult: 带有确切结果与成功与否标识体，支持大体积落盘记录。
        """
        # --- 1. 准备上下文并验证 ---
        context.call_id = tool_use.id
        context.tool_name = tool_use.name
        execution_context = context
        tool: BaseTool | None = None
        try:
            tool = self.registry.get(tool_use.name)
            if self.circuit_breaker is not None:
                try:
                    self.circuit_breaker.before_call(tool.name)
                except IrisToolExecutionError as exc:
                    return self._error_result(
                        tool_use,
                        str(exc.context.get("code", "CIRCUIT_OPEN")),
                        exc.message,
                        details=exc.context,
                    )
            params = tool.validate_input(tool_use.input)
            raw_params = params.model_dump() if isinstance(params, BaseModel) else dict(params)
            middleware_error = await self._run_before_call(tool, raw_params, execution_context)
            if middleware_error is not None:
                self._record_breaker_result(tool.name, middleware_error)
                return middleware_error

            # --- 2. 执行权限策略 ---
            try:
                decision = self.permission_policy.check(tool, raw_params, execution_context)
            except Exception as exc:
                result = self._error_result(tool_use, "PERMISSION_ERROR", str(exc))
                self._record_breaker_result(tool.name, result)
                return result

            if not decision.allowed:
                result = self._error_result(
                    tool_use,
                    "PERMISSION_ERROR",
                    decision.reason,
                    details={
                        "require_confirmation": decision.require_confirmation,
                        **decision.metadata,
                    },
                )
                self._record_breaker_result(tool.name, result)
                return result

            # --- 3. 执行工具并格式化结果 ---
            try:
                result = await tool.arun(params, execution_context)
            except Exception as exc:
                handled = await self._run_on_error(tool, exc, execution_context)
                if handled is None:
                    raise
                result = handled
            normalized = result.model_copy(
                update={
                    "tool_use_id": result.tool_use_id or tool_use.id,
                    "tool_name": result.tool_name or tool_use.name,
                }
            )

            # --- 4. 若数据量较大，处理持久化存储 ---
            artifact_store = self._artifact_store(execution_context)
            persisted = artifact_store.persist_if_large(
                normalized,
                max_chars=tool.definition.max_result_chars,
            )
            final_result = await self._run_after_call(tool, persisted, execution_context)
            final_result = final_result.model_copy(
                update={
                    "tool_use_id": final_result.tool_use_id or tool_use.id,
                    "tool_name": final_result.tool_name or tool_use.name,
                }
            )
            self._record_breaker_result(tool.name, final_result)
            return final_result
        except IrisToolNotFoundError:
            return self._error_result(tool_use, "NOT_FOUND", f"工具不存在: {tool_use.name}")
        except (IrisToolValidationError, ValidationError) as exc:
            result = self._error_result(tool_use, "VALIDATION_ERROR", str(exc))
            if tool is not None:
                self._record_breaker_result(tool.name, result)
            return result
        except IrisToolExecutionError as exc:
            allow_structured = tool is not None and (
                tool.definition.group == "file" or exc.message.startswith("ARTIFACT_ERROR:")
            )
            code, message = _tool_error_code_and_message(
                exc.message,
                allow_structured=allow_structured,
            )
            result = self._error_result(tool_use, code, message)
            if tool is not None:
                self._record_breaker_result(tool.name, result)
            return result
        except Exception as exc:
            result = self._error_result(tool_use, "EXECUTION_ERROR", str(exc))
            if tool is not None:
                self._record_breaker_result(tool.name, result)
            return result

    async def execute_many(
        self,
        tool_uses: Sequence[ToolUseBlock],
        context: ToolExecutionContext,
    ) -> list[ToolResult]:
        """执行多个工具调用，只读且并发安全的连续批次并发执行。

        识别指令清单，自动收集处于可以只读的工具做并行调度从而节约周期；
        遇到写状态改变等风险工具立刻退化为阻塞调用以保证顺序。

        Args:
            tool_uses (Sequence[ToolUseBlock]): 从模型抽取的当次循环执行清单集。
            context (ToolExecutionContext): 供当次工具使用的透传依赖。

        Returns:
            list[ToolResult]: 所有结果合包，结果序列一一对应清单集原有次序。
        """
        results: list[ToolResult] = []
        batch: list[ToolUseBlock] = []
        for tool_use in tool_uses:
            if self._is_read_only_concurrency_safe(tool_use):
                batch.append(tool_use)
                continue
            if batch:
                results.extend(await self._execute_read_batch(batch, context))
                batch = []
            results.append(await self.execute_one(tool_use, context))
        if batch:
            results.extend(await self._execute_read_batch(batch, context))
        return results

    # endregion

    # ==========================================
    #               Helper Methods
    # ==========================================
    # region
    def _error_result(
        self,
        tool_use: ToolUseBlock,
        code: str,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> ToolResult:
        """构造错误工具结果。

        作为包装错误栈、异常文本到统一响应对象出口的方法，保证模型侧认知规范。

        Args:
            tool_use (ToolUseBlock): 引发异常的原请求信息。
            code (str): 大写带下划线的标准类型标记。
            message (str): 描述异常情形的详细反馈体。
            details (dict[str, object] | None): 附加可能存在的部分详细风控上下文。

        Returns:
            ToolResult: is_error 生效情况下的专供结构体。
        """
        return ToolResult(
            tool_use_id=tool_use.id,
            tool_name=tool_use.name,
            is_error=True,
            error=ToolErrorInfo(code=code, message=message, details=details or {}),
        )

    async def _execute_read_batch(
        self,
        tool_uses: list[ToolUseBlock],
        context: ToolExecutionContext,
    ) -> list[ToolResult]:
        """并发执行连续只读批次。

        将一串经过判断安全的调用批量压入异步调度池内争取快速返回。

        Args:
            tool_uses (list[ToolUseBlock]): 需要同步执行的任务组合。
            context (ToolExecutionContext): 生命周期环境。

        Returns:
            list[ToolResult]: 生成的已完成数据流集。
        """
        tasks = (
            self.execute_one(tool_use, context.model_copy(deep=True)) for tool_use in tool_uses
        )
        return list(await asyncio.gather(*tasks))

    def _is_read_only_concurrency_safe(self, tool_use: ToolUseBlock) -> bool:
        """判断工具调用是否可进入只读并发批次。

        通过请求注册表的定义对具体指令及装载参数判断并行支持兼容度。

        Args:
            tool_use (ToolUseBlock): 测试能否无锁快速通行的请求件。

        Returns:
            bool: 回报其支持策略状态。有验证崩溃或缺失统统退库阻塞排队。
        """
        try:
            tool = self.registry.get(tool_use.name)
            params = tool.validate_input(tool_use.input)
            raw_params = params.model_dump() if isinstance(params, BaseModel) else dict(params)
            return tool.is_read_only(raw_params) and tool.is_concurrency_safe(raw_params)
        except (IrisToolNotFoundError, IrisToolValidationError, ValidationError):
            return False

    def _artifact_store(self, context: ToolExecutionContext) -> ToolArtifactStore:
        """为当前上下文创建 artifact store。

        获取指向缓存存盘对应工作目内配置的实例，管理超大尺寸响应信息。

        Args:
            context (ToolExecutionContext): 提供执行标识路径信息的宿主。

        Returns:
            ToolArtifactStore: 操作落盘工作的具象存取处理库。
        """
        session_id = _safe_path_segment(context.session_id or "default")
        root = context.workspace_root / ".iris" / "tool-results" / session_id
        return ToolArtifactStore(root=root, preview_chars=self.artifact_preview_chars)

    async def _run_before_call(
        self,
        tool: BaseTool,
        params: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult | None:
        """运行 before_call middleware。"""
        for middleware in self.middleware:
            hook = getattr(middleware, "before_call", None)
            if hook is None:
                continue
            try:
                await _maybe_await(hook(tool, params, context))
            except Exception as exc:
                return self._error_result(
                    _tool_use_from_context(context),
                    "MIDDLEWARE_ERROR",
                    str(exc),
                )
        return None

    async def _run_after_call(
        self,
        tool: BaseTool,
        result: ToolResult,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """运行 after_call 和兼容 after_execute middleware。"""
        current = result
        for middleware in self.middleware:
            try:
                hook = getattr(middleware, "after_call", None)
                if hook is not None:
                    current = await _maybe_await(hook(tool, current, context))
                legacy_hook = getattr(middleware, "after_execute", None)
                if legacy_hook is not None:
                    current = await _maybe_await(legacy_hook(current, context))
            except Exception as exc:
                return self._error_result(
                    _tool_use_from_context(context),
                    "MIDDLEWARE_ERROR",
                    str(exc),
                )
        return current

    async def _run_on_error(
        self,
        tool: BaseTool,
        error: Exception,
        context: ToolExecutionContext,
    ) -> ToolResult | None:
        """运行 on_error middleware。"""
        for middleware in self.middleware:
            hook = getattr(middleware, "on_error", None)
            if hook is None:
                continue
            try:
                replacement = await _maybe_await(hook(tool, error, context))
            except Exception as exc:
                return self._error_result(
                    _tool_use_from_context(context),
                    "MIDDLEWARE_ERROR",
                    str(exc),
                    details={
                        "original_error": str(error),
                        "middleware_error": str(exc),
                    },
                )
            if replacement is not None:
                return replacement
        return None

    def _record_breaker_result(self, tool_name: str, result: ToolResult) -> None:
        """将执行结果写入熔断器。"""
        if self.circuit_breaker is not None:
            self.circuit_breaker.after_result(tool_name, result)

    # endregion


def _tool_error_code_and_message(
    message: str,
    *,
    allow_structured: bool,
) -> tuple[str, str]:
    """从工具异常消息中提取稳定错误码。

    解析系统抛出的不规则报错短语，将其按系统规格分离为大写键和文字段的结构。

    Args:
        message (str): 捕获得到的待分析文本。
        allow_structured (bool): 是否允许对其执行结构化键值分析，否则直接按明文输出。

    Returns:
        tuple[str, str]: 可直接作为字典键值存储的安全双元组。
    """
    if not allow_structured:
        return "EXECUTION_ERROR", message
    match = re.match(r"^([A-Z][A-Z0-9_]+):\s*(.*)$", message)
    if match is None:
        return "EXECUTION_ERROR", message
    return match.group(1), match.group(2)


def _safe_path_segment(value: str) -> str:
    """将外部 ID 转为单个安全路径段。

    确保所生成字符片段能够跨系统文件层安全存储，清除所有特殊符号。

    Args:
        value (str): 要清理保护的文件节点命名。

    Returns:
        str: 规整并替代好禁用位后的合法纯字符串。
    """
    segment = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    return segment.strip("_") or "default"


async def _maybe_await(value: Awaitable[Any] | Any) -> Any:
    """兼容同步和异步 middleware 返回值。"""
    if inspect.isawaitable(value):
        return await value
    return value


def _tool_use_from_context(context: ToolExecutionContext) -> ToolUseBlock:
    """用上下文构造错误结果所需的工具调用占位。"""
    return ToolUseBlock(id=context.call_id, name=context.tool_name, input={})
