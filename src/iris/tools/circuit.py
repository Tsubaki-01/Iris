"""工具调用熔断器。"""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel

from ..exceptions import IrisToolExecutionError
from .base import ToolResult


class CircuitBreakerState(BaseModel):
    """单个工具的熔断状态。"""

    tool_name: str
    failure_count: int = 0
    opened_at: datetime | None = None
    last_error_code: str = ""


class CircuitBreaker:
    """阻止持续失败的工具被反复调用。"""

    def __init__(
        self, *, failure_threshold: int = 3, cooldown_seconds: float = 30.0
    ) -> None:
        """初始化熔断器。

        Args:
            failure_threshold (int): 连续失败达到该数量后打开熔断。
            cooldown_seconds (float): 打开后允许再次试探的冷却秒数。
        """
        self.failure_threshold = max(failure_threshold, 1)
        self.cooldown = timedelta(seconds=max(cooldown_seconds, 0.0))
        self._states: dict[str, CircuitBreakerState] = {}

    def before_call(self, tool_name: str) -> None:
        """检查工具是否处于熔断状态。

        Args:
            tool_name (str): 工具名称。

        Raises:
            IrisToolExecutionError: 熔断仍打开时抛出。
        """
        state = self._states.get(tool_name)
        if state is None or state.opened_at is None:
            return
        if datetime.now() - state.opened_at >= self.cooldown:
            state.opened_at = None
            return
        raise IrisToolExecutionError(
            "工具暂时熔断，稍后重试",
            tool_name=tool_name,
            code="CIRCUIT_OPEN",
            last_error_code=state.last_error_code,
        )

    def after_result(self, tool_name: str, result: ToolResult) -> None:
        """根据工具结果更新失败计数。

        Args:
            tool_name (str): 工具名称。
            result (ToolResult): 本次执行结果。
        """
        if not result.is_error:
            self.reset(tool_name)
            return
        state = self._states.setdefault(
            tool_name, CircuitBreakerState(tool_name=tool_name)
        )
        state.failure_count += 1
        state.last_error_code = result.error.code if result.error is not None else ""
        if state.failure_count >= self.failure_threshold:
            state.opened_at = datetime.now()

    def reset(self, tool_name: str) -> None:
        """清除指定工具的熔断状态。"""
        self._states.pop(tool_name, None)
