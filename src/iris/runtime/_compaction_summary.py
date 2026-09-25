"""摘要历史的完整序列化、动态分批与纯文本响应消费。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..agents.config.compaction import CompactionConfig
from ..exceptions import IrisContextCompactionError
from ..message import LLMRequest, LLMResponse, Msg, TextBlock, ToolUseBlock
from ..utils import TemplateRenderer
from ._prompts import render_prompt

_INPUT_TEMPLATE = Path(__file__).parents[1] / "prompts" / "compaction_input.j2"


@dataclass(frozen=True, slots=True)
class SummaryRecord:
    """一个有稳定身份的历史内容块；正文允许跨批次按字符分片。"""

    header: str
    text: str


@dataclass(frozen=True, slots=True)
class SummaryBatch:
    """一个满足输入预算的请求及下一段未消费位置。"""

    request: LLMRequest
    next_position: tuple[int, int]


def serialize_history(messages: list[Msg], start_index: int) -> tuple[SummaryRecord, ...]:
    """按原始消息和内容块顺序保留摘要所需的全部正文及工具事实。"""
    records: list[SummaryRecord] = []
    for message_index, message in enumerate(messages, start=start_index):
        identity = (
            f"message={message_index} | ref=message:{message_index} | role={message.role.value} | "
            f"sender={json.dumps(message.sender, ensure_ascii=False)}"
        )
        if isinstance(message.content, str) or not message.content:
            text = message.content if isinstance(message.content, str) else ""
            records.append(SummaryRecord(f"{identity} | block=0 | kind=text", text))
            continue
        for block_index, block in enumerate(message.content):
            header = f"{identity} | block={block_index}"
            if isinstance(block, TextBlock):
                records.append(SummaryRecord(f"{header} | kind=text", block.text))
            elif isinstance(block, ToolUseBlock):
                records.append(
                    SummaryRecord(
                        f"{header} | kind=tool_call | "
                        f"call_id={json.dumps(block.id, ensure_ascii=False)} | "
                        f"name={json.dumps(block.name, ensure_ascii=False)} | "
                        "execution_status=issued",
                        "arguments: "
                        + json.dumps(block.input, ensure_ascii=False, separators=(",", ":")),
                    )
                )
            else:
                status = "failed" if block.is_error else "completed"
                text = block.content
                for key in ("error", "artifact"):
                    if key in block.metadata:
                        text += f"\n{key}: " + json.dumps(
                            block.metadata[key], ensure_ascii=False, separators=(",", ":")
                        )
                records.append(
                    SummaryRecord(
                        f"{header} | ref=result:{message_index}:{block_index} | kind=tool_result | "
                        f"call_id={json.dumps(block.tool_use_id, ensure_ascii=False)} | "
                        f"name={json.dumps(block.name, ensure_ascii=False)} | "
                        f"execution_status={status} | is_error={str(block.is_error).lower()}",
                        text,
                    )
                )
    return tuple(records)


def next_summary_batch(
    main_request: LLMRequest,
    previous_summary: str | None,
    records: tuple[SummaryRecord, ...],
    position: tuple[int, int],
    config: CompactionConfig,
    estimate_input_tokens: Callable[[LLMRequest], int],
    *,
    system_prompt: str,
    prompt_renderer: TemplateRenderer,
) -> SummaryBatch:
    """指数探测完整记录前缀，再按需二分细化和切分长正文。

    每个返回请求都经过完整计量；不要求找到数学上的最大装填前缀。
    调用方仅在仍有未处理记录时调用；一批响应成功后才承接返回位置。
    """
    fragments: list[str] = []
    index, offset = position
    remaining = len(records) - index

    def build(parts: list[str]) -> LLMRequest:
        return _summary_request(
            main_request,
            previous_summary,
            "\n\n".join(parts),
            config,
            system_prompt,
            prompt_renderer,
        )

    def prefix(count: int) -> LLMRequest:
        # 只渲染已经探测到的记录，避免每批先物化全部未消费历史。
        while len(fragments) < count:
            record = records[index + len(fragments)]
            start = offset if not fragments else 0
            fragments.append(_render_fragment(record, start, len(record.text)))
        return build(fragments[:count])

    selected = 0
    upper = 1
    request: LLMRequest | None = None
    while True:
        candidate = prefix(upper)
        if estimate_input_tokens(candidate) <= config.input_budget_tokens:
            selected = upper
            request = candidate
            if selected == remaining:
                return SummaryBatch(request, (len(records), 0))
            upper = min(remaining, upper * 2)
        else:
            break

    lower, upper = selected + 1, upper - 1
    while lower <= upper:
        midpoint = (lower + upper) // 2
        candidate = prefix(midpoint)
        if estimate_input_tokens(candidate) <= config.input_budget_tokens:
            selected = midpoint
            request = candidate
            lower = midpoint + 1
        else:
            upper = midpoint - 1

    # 完整记录放不下时，只搜索下一条的剩余正文；失败候选不覆盖成功请求。
    index += selected
    offset = offset if selected == 0 else 0
    record = records[index]
    lower, upper = offset + 1, len(record.text) - 1
    next_offset = offset
    while lower <= upper:
        midpoint = (lower + upper) // 2
        candidate = build([*fragments[:selected], _render_fragment(record, offset, midpoint)])
        if estimate_input_tokens(candidate) <= config.input_budget_tokens:
            next_offset = midpoint
            request = candidate
            lower = midpoint + 1
        else:
            upper = midpoint - 1
    if request is not None:
        return SummaryBatch(request, (index, next_offset))
    raise IrisContextCompactionError(
        "摘要指令、工作摘要与最小历史片段无法装入输入预算",
        code="CONTEXT_COMPACTION_UNAVAILABLE",
    )


def consume_summary_response(response: LLMResponse) -> str:
    """在 usage 已保存后，只接收完整、非空且无工具块的文本摘要。"""
    if response.finish_reason != "stop" or any(
        not isinstance(block, TextBlock) for block in response.content
    ):
        raise IrisContextCompactionError(
            "摘要响应必须完整结束且只包含文本", code="CONTEXT_COMPACTION_FAILED"
        )
    summary = "\n".join(block.text for block in response.content).strip()
    if not summary:
        raise IrisContextCompactionError("摘要响应为空", code="CONTEXT_COMPACTION_FAILED")
    return summary


def _render_fragment(record: SummaryRecord, start: int, end: int) -> str:
    coverage = "complete" if start == 0 and end == len(record.text) else "partial"
    return (
        f"[record {record.header} | "
        f"text_coverage={coverage} [{start},{end})/{len(record.text)}]\n"
        f"{record.text[start:end]}"
    )


def _summary_request(
    main_request: LLMRequest,
    previous_summary: str | None,
    serialized_history: str,
    config: CompactionConfig,
    system_prompt: str,
    prompt_renderer: TemplateRenderer,
) -> LLMRequest:
    return main_request.model_copy(
        update={
            "messages": [
                Msg.system(system_prompt),
                Msg.user(
                    render_prompt(
                        prompt_renderer,
                        _INPUT_TEMPLATE,
                        {
                            "previous_summary_or_none": (
                                "(none)" if previous_summary is None else previous_summary
                            ),
                            "serialized_history": serialized_history,
                        },
                    )
                ),
            ],
            "stream": False,
            "tools": [],
            "tool_choice": None,
            "response_format": None,
            "max_tokens": config.summary_tokens,
            "provider_options": {**main_request.provider_options, "num_retries": 0},
        }
    )
