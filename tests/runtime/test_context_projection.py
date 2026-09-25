"""重复结果与旧观察正文仅改变当前模型请求。"""

import json
from collections.abc import Callable

import pytest

from iris.agents import ContextPolicyConfig
from iris.lifecycle import SessionCompaction
from iris.message import LLMRequest, Msg, TextBlock, ToolResultBlock, ToolUseBlock
from iris.runtime._context_projection import prune_tool_results
from iris.runtime.compaction import project_history


def _batch(
    text: str,
    *,
    name: str = "read",
    call_id: str = "same",
    args: dict | None = None,
    retention: str = "observation",
    error: bool = False,
    artifact: bool = False,
) -> list[Msg]:
    """构造一个已闭合批次，保留 executor 固化的声明事实。"""
    metadata = {"extra": {"context_retention": retention, "context_tool_name": "canonical_read"}}
    if artifact:
        metadata["artifact"] = {"path": "/different-at-runtime.txt"}
    return [
        Msg.assistant([ToolUseBlock(id=call_id, name=name, input=args or {"path": "x"})]),
        Msg.tool_result(
            tool_use_id=call_id, name=name, content=text, is_error=error, metadata=metadata
        ),
    ]


def _estimate(request: LLMRequest) -> int:
    """计量整包选项、schema 和消息文本。"""
    return len(json.dumps(request.tools)) + sum(
        len(
            block.content
            if isinstance(block, ToolResultBlock)
            else block.text
            if isinstance(block, TextBlock)
            else json.dumps(block.input)
        )
        for message in request.messages
        for block in message.blocks
    )


def _project(
    messages: list[Msg],
    *,
    trigger: int = 100,
    recent: int = 2,
    preview: int = 16,
    choice: str | dict | None = None,
    tools: bool = True,
    estimate: Callable[[LLMRequest], int] = _estimate,
) -> LLMRequest:
    request = LLMRequest(
        model="test",
        messages=[Msg.system("rules"), *messages],
        tools=[{"type": "function", "function": {"name": "context_read"}}] if tools else [],
        tool_choice=choice,
    )
    return prune_tool_results(
        request,
        source_indices={id(message): i for i, message in enumerate(messages)},
        config=ContextPolicyConfig(
            preserve_recent_tool_groups=recent, old_result_preview_chars=preview
        ),
        trigger_tokens=trigger,
        estimate_input_tokens=estimate,
    )


def test_exact_duplicates_keep_real_pairs_recent_batches_and_raw() -> None:
    """三次同结果只折叠最早正文，真实调用对和后两批原文保留。"""
    body = "unchanged observation " * 100
    raw = [*_batch(body), *_batch(body), *_batch(body)]
    before = [message.model_dump() for message in raw]
    projected = _project(raw)
    results = [block for message in projected.messages for block in message.tool_results]
    assert "重复正文见 result:5:0" in results[0].content
    assert "本次原文 result:1:0" in results[0].content
    assert [block.content for block in results[1:]] == [body, body]
    assert sum(len(message.tool_calls) for message in projected.messages) == 3
    assert len(results) == 3
    assert [message.model_dump() for message in raw] == before


def test_aliases_use_saved_canonical_name_and_json_key_order() -> None:
    """不同 alias 与参数 key 顺序不改变实际同一观察的判等。"""
    body = "same" * 500
    projected = _project(
        [
            *_batch(body, name="alias_a", args={"a": 1, "b": 2}),
            *_batch(body, name="alias_b", args={"b": 2, "a": 1}),
        ],
        recent=0,
    )
    assert "重复正文见 result:3:0" in projected.messages[2].tool_results[0].content
    assert projected.messages[4].tool_results[0].content == body


@pytest.mark.parametrize("kind", ["changed", "artifact", "keep", "error", "open"])
def test_nonduplicates_and_protected_results_do_not_fold(kind: str) -> None:
    """同参数不等于同结果，产物预览和不可裁剪事实不折叠。"""
    body = "observation" * 100
    first = _batch(
        body,
        artifact=kind == "artifact",
        retention="keep" if kind == "keep" else "observation",
        error=kind == "error",
    )
    last = _batch(body + ("new" if kind == "changed" else ""))
    if kind == "open":
        first[0] = Msg.assistant([*first[0].tool_calls, ToolUseBlock(id="pending", name="read")])
        last = []
    raw = [*first, *last]
    projected = _project(raw, recent=0, preview=10000)
    assert all(
        "重复正文见" not in block.content
        for message in projected.messages
        for block in message.tool_results
    )
    assert [message.model_dump() for message in projected.messages[1:]] == [
        message.model_dump() for message in raw
    ]


def test_unique_old_results_shorten_but_actual_duplicate_representative_stays() -> None:
    """唯一旧观察可以短化，实际重复代表仍提供完整正文。"""
    duplicate, unique = "duplicate" * 200, "HEAD" + "middle" * 300 + "TAIL"
    raw = [*_batch(duplicate), *_batch(duplicate), *_batch(unique)]
    projected = _project(raw, recent=0, preview=8)
    results = [block for message in projected.messages for block in message.tool_results]
    assert "重复正文见" in results[0].content
    assert results[1].content == duplicate
    assert "原文：result:5:0" in results[2].content
    assert "预览：HEADmi" in results[2].content and results[2].content.endswith("IL")


@pytest.mark.parametrize(
    "choice,tools",
    [(None, False), ("none", True), ({"type": "function", "function": {"name": "other"}}, True)],
)
def test_unavailable_context_read_keeps_all_bodies(choice: str | dict | None, tools: bool) -> None:
    raw = [*_batch("body" * 500), *_batch("body" * 500)]
    assert _project(raw, recent=0, choice=choice, tools=tools).messages[1:] == raw


def test_low_pressure_and_token_nonbenefit_keep_request() -> None:
    raw = [*_batch("body" * 500), *_batch("body" * 500)]
    assert _project(raw, recent=0, trigger=10000).messages[1:] == raw
    assert _project(raw, recent=0, estimate=lambda request: 1000).messages[1:] == raw


def test_each_compaction_candidate_recomputes_representatives_at_original_indices() -> None:
    """候选只剩唯一结果时解除代表保护，但其 ref 仍是原始位置。"""
    body = "unchanged" * 200
    raw = [*_batch(body), *_batch(body), *_batch(body)]
    indices = {id(message): index for index, message in enumerate(raw)}

    def candidate(end: int) -> LLMRequest:
        history = project_history(
            raw, SessionCompaction(summary="summary", covered_message_count=end), ()
        )
        return prune_tool_results(
            LLMRequest(
                model="test", messages=history, tools=[{"function": {"name": "context_read"}}]
            ),
            source_indices=indices,
            config=ContextPolicyConfig(preserve_recent_tool_groups=0, old_result_preview_chars=0),
            trigger_tokens=100,
            estimate_input_tokens=_estimate,
        )

    earlier = candidate(2)
    assert "重复正文见 result:5:0" in earlier.messages[2].tool_results[0].content
    assert earlier.messages[4].tool_results[0].content == body
    later = candidate(4)
    assert "原文：result:5:0" in later.messages[2].tool_results[0].content
    assert "重复正文见" not in later.messages[2].tool_results[0].content
    assert raw[5].tool_results[0].content == body


def test_recent_window_counts_complete_parallel_batches_not_individual_results() -> None:
    """双工具批次整体保护，普通对话不消耗近期工具组名额。"""
    body = "old observation" * 100
    parallel = [
        Msg.assistant(
            [
                ToolUseBlock(id="a", name="read", input={"path": "x"}),
                ToolUseBlock(id="b", name="read", input={"path": "x"}),
            ]
        ),
        _batch(body, call_id="a")[1],
        _batch(body, call_id="b")[1],
    ]
    raw = [*_batch(body), *parallel, Msg.user("new instructions"), *_batch("fresh" * 300)]
    projected = _project(raw)
    assert "重复正文见" in projected.messages[2].tool_results[0].content
    assert projected.messages[4].tool_results[0].content == body
    assert projected.messages[5].tool_results[0].content == body
