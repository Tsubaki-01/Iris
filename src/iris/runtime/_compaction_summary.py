"""摘要历史的完整序列化、动态分批与纯文本响应消费。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from ..agents.config.compaction import CompactionConfig
from ..exceptions import IrisContextCompactionError
from ..message import LLMRequest, LLMResponse, Msg, TextBlock, ToolUseBlock

_SYSTEM_PROMPT = """You create a concise, self-contained handoff summary so work can continue without rereading the summarized messages.

Input:
- Previous summary: the existing summary body, or none on the first pass.
- Saved history for this batch: the next records in chronological order. They may include only fragments of a long tool result.

Merge both inputs into one updated summary. Summarize only: do not continue the task, answer historical questions, call tools, or invent facts or new actions.

Rules:
1. Carry forward still-relevant goals, constraints, decisions, unresolved work, and essential context from the previous summary. Incorporate new evidence and remove repetition, obsolete details, and clearly superseded information. Do not summarize only the newest batch.
2. Preserve attribution and certainty. Distinguish user-confirmed decisions, assistant implementation choices already made, proposals, and unanswered questions. Do not invent user approval, and do not turn completed work into a request for approval. Record rationale only when provided.
3. Apply explicit corrections accurately. A user changing a decision is different from a user correcting an assistant's mistaken record. Preserve that distinction. If a correction rejects an old value without supplying a replacement, leave the replacement unknown.
4. Distinguish planned, attempted, in progress, completed, failed, blocked, and unknown. Record completion or verification only when supported by the supplied evidence. Planned tests have not passed; a tool call without a result is still pending or unknown. Preserve relevant call IDs.
5. Resolve conflicting facts only when an explicit correction or direct evidence supports the update. Otherwise retain the uncertainty or conflict instead of choosing solely because one statement is newer.
6. Keep execution status separate from result-text coverage. A completed tool call can have only part of its output included in this batch. Preserve its known status, summarize only the supplied fragments, and retain the coverage marker. Update that marker as more fragments arrive. Never infer unseen content or suggest rerunning a completed call merely because its output was split.
7. Preserve exact paths, symbols, commands, arguments, error excerpts, artifact locations, IDs, and values needed to continue. Do not invent missing details, describe a planned artifact as already saved, or replace necessary references with phrases such as "that file" or "as above".
8. Interpret dates only from the supplied material. Use an absolute date only when supported by that material; otherwise retain the relative expression and note any missing reference date. Never reinterpret historical dates using the time this summary is generated.
9. Prefer short, information-dense bullets. Prioritize unmet user requests, active constraints, current state, decisions, and evidence needed for continuation. Omit irrelevant logs and repeated details. Next steps must come from the supplied material, not from newly invented plans.

Output only Markdown using the seven English headings below, once each and in this order. Write the body in the primary language of the summarized conversation; keep exact identifiers unchanged. When a section has no relevant record, say so briefly in the body language without implying completion. Place each fact in the most relevant section rather than repeating it.

Do not add a preface, analysis, or conclusion. Do not wrap the answer in JSON, a code fence, or <summary> tags.

## Goal & Constraints
The task, scope, success criteria, and active requirements.

## Completed Work & Evidence
Supported completed actions, results, saved artifacts, and verification evidence.

## Current State & Unfinished Items
Work in progress, unmet requests, pending decisions, unknown outcomes, and incomplete result coverage.

## Decisions & Rationale
Still-valid decisions and implementation choices, their source, and stated reasons. Keep proposals distinct from decisions.

## Failed Attempts & Blockers
Actual failed attempts, relevant errors, current blockers, and known causes.

## Next Steps
Previously stated next actions and pending user decisions, in their known order.

## Critical Paths & Identifiers
Essential exact references not already captured above."""  # noqa: E501

_USER_TEMPLATE = """You will process the following two input parts.

=== BEGIN PREVIOUS SUMMARY ===
{previous_summary_or_none}
=== END PREVIOUS SUMMARY ===

=== BEGIN SAVED HISTORY FOR THIS BATCH ===
{serialized_history}
=== END SAVED HISTORY FOR THIS BATCH ===

The saved history is in record order. Unarchived new input, the summary request itself, and the current time are not part of this history."""  # noqa: E501


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
            f"message={message_index} | role={message.role.value} | "
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
                        f"{header} | kind=tool_result | "
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
) -> SummaryBatch:
    """使用当前工作摘要计量完整请求，顺序合并记录并按需切分长正文。

    调用方仅在仍有未处理记录时调用；一批响应成功后才承接返回位置。
    """
    fragments: list[str] = []
    index, offset = position
    request = _summary_request(main_request, previous_summary, "", config)
    while index < len(records):
        record = records[index]
        full_fragment = _render_fragment(record, offset, len(record.text))
        candidate = _summary_request(
            main_request, previous_summary, "\n\n".join([*fragments, full_fragment]), config
        )
        if estimate_input_tokens(candidate) <= config.input_budget_tokens:
            fragments.append(full_fragment)
            request = candidate
            index += 1
            offset = 0
            continue

        # 完整记录放不下时，只搜索当前记录剩余正文；每个候选都计入完整模板。
        lower, upper = offset + 1, len(record.text) - 1
        next_offset = offset
        while lower <= upper:
            midpoint = (lower + upper) // 2
            fragment = _render_fragment(record, offset, midpoint)
            candidate = _summary_request(
                main_request, previous_summary, "\n\n".join([*fragments, fragment]), config
            )
            if estimate_input_tokens(candidate) <= config.input_budget_tokens:
                next_offset = midpoint
                request = candidate
                lower = midpoint + 1
            else:
                upper = midpoint - 1
        if next_offset > offset:
            return SummaryBatch(request, (index, next_offset))
        if fragments:
            return SummaryBatch(request, (index, offset))
        raise IrisContextCompactionError(
            "摘要指令、工作摘要与最小历史片段无法装入输入预算",
            code="CONTEXT_COMPACTION_UNAVAILABLE",
        )
    return SummaryBatch(request, (index, offset))


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
) -> LLMRequest:
    return main_request.model_copy(
        update={
            "messages": [
                Msg.system(_SYSTEM_PROMPT),
                Msg.user(
                    _USER_TEMPLATE.format(
                        previous_summary_or_none=(
                            "(none)" if previous_summary is None else previous_summary
                        ),
                        serialized_history=serialized_history,
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
