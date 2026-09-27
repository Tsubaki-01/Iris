"""让精确请求与其完整输入计量一起流转。"""

from collections.abc import Callable
from dataclasses import dataclass

from ..message import LLMRequest


@dataclass(frozen=True, slots=True)
class MeasuredRequest:
    """已完成计量的请求；正文和选项变化时必须构造并计量新候选。"""

    request: LLMRequest
    input_tokens: int


def measure_request(
    request: LLMRequest, estimate_input_tokens: Callable[[LLMRequest], int]
) -> MeasuredRequest:
    """对当前完整请求计量一次，并保留对应请求。"""
    return MeasuredRequest(request, estimate_input_tokens(request))
