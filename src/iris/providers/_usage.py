"""保留服务商实际报告的 token 字段，不把未知计数补成已知零。"""

from collections.abc import Mapping
from typing import Any

from ..message import LLMResponse, ModelUsageSnapshot


def parse_usage(usage: Mapping[str, Any], *, chat_completions: bool = False) -> dict[str, int]:
    """将原始非 null 计数转换为 Iris 字段，保留服务商报告的零。"""
    fields = (
        ("input_tokens", "prompt_tokens" if chat_completions else "input_tokens"),
        ("output_tokens", "completion_tokens" if chat_completions else "output_tokens"),
        ("total_tokens", "total_tokens"),
    )
    return {
        target: int(value) for target, source in fields if (value := usage.get(source)) is not None
    }


def known_usage(usage: LLMResponse | ModelUsageSnapshot) -> dict[str, int]:
    """在响应与快照之间投影已知字段，不改变业务默认计数。"""
    return {
        name: value
        for name, value in (
            ("input_tokens", usage.input_tokens),
            ("output_tokens", usage.output_tokens),
            ("total_tokens", usage.total_tokens),
        )
        if name in usage.model_fields_set
    }
