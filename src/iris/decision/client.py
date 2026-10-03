"""业务消费者和宿主替身共用的窄判断协议。"""

from typing import Protocol

from .models import DecisionRequest, DecisionResponse


class DecisionEvaluator(Protocol):
    """只提供一次批量判断，不向借用方暴露资源关闭职责。"""

    async def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        """返回本请求全部问题的可信答案。"""
        ...


__all__ = ["DecisionEvaluator"]
