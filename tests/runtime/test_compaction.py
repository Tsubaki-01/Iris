"""历史压缩的原文保护、投影与合法切点测试。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from fakes import history_snapshot

from iris.agents import CompactionConfig
from iris.lifecycle import SessionCompaction
from iris.message import (
    ImageBlock,
    ImageFileRef,
    LLMRequest,
    Msg,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from iris.runtime._request_measurement import MeasuredRequest, measure_request
from iris.runtime.compaction import (
    project_history,
    select_compaction_end,
)


def _request(history: list[Msg]) -> LLMRequest:
    return LLMRequest(model="test-model", messages=[Msg.system("rules"), *history])


def _estimate(request: LLMRequest) -> int:
    count = 0
    for message in request.messages:
        count += 4
        for block in message.blocks:
            if isinstance(block, TextBlock):
                count += len(block.text)
            elif isinstance(block, ToolUseBlock):
                count += len(block.id) + len(block.name) + len(json.dumps(block.input))
            elif isinstance(block, ToolResultBlock):
                count += len(block.tool_use_id) + len(block.text)
    count += len(json.dumps([tool.model_dump() for tool in request.tools])) if request.tools else 0
    count += len(json.dumps(request.response_format)) if request.response_format else 0
    return count


def _select(
    messages: list[Msg],
    *,
    initial_count: int = 0,
    previous: SessionCompaction | None = None,
    config: CompactionConfig | None = None,
    build_request: Callable[[list[Msg]], LLMRequest] = _request,
) -> int | None:
    return select_compaction_end(
        snapshot=history_snapshot(messages, initial_count=initial_count, compaction=previous),
        config=config or CompactionConfig(input_budget_tokens=1000),
        build_request=lambda history: measure_request(build_request(history), _estimate),
    )


def test_protects_current_run_bci_input_and_metadata_free_latest_steer() -> None:
    messages = [
        Msg.user("旧环境", sender="context", metadata={"context_kind": "before_current_input"}),
        Msg.user("同一文本"),
        Msg.assistant("旧回复"),
        Msg.user("当前环境", sender="context", metadata={"context_kind": "before_current_input"}),
        Msg.user("同一文本"),
        Msg.assistant("正在处理"),
        Msg.user("早先方向"),
        Msg.user("新方向"),
        Msg.tool_result(tool_use_id="call", content="结果"),
    ]

    assert history_snapshot(messages, initial_count=3).protected_indices == (3, 4, 7)


def test_current_run_without_bci_never_borrows_old_context() -> None:
    messages = [
        Msg.user("旧环境", sender="context", metadata={"context_kind": "before_current_input"}),
        Msg.user("旧任务"),
        Msg.assistant("完成"),
        Msg.user("新任务"),
        Msg.assistant("处理中"),
    ]

    assert history_snapshot(messages, initial_count=3).protected_indices == (3,)
    assert history_snapshot(messages, initial_count=len(messages)).protected_indices == ()


def test_projection_restores_covered_anchors_once_in_original_order() -> None:
    messages = [
        Msg.user("旧历史"),
        Msg.user("当前环境", sender="context", metadata={"context_kind": "before_current_input"}),
        Msg.user("当前任务"),
        Msg.assistant("第一步"),
        Msg.user("最新方向"),
        Msg.assistant("第二步"),
    ]
    compacted = SessionCompaction(summary="工作摘要", covered_message_count=4)

    snapshot = history_snapshot(messages, initial_count=1, compaction=compacted)
    projected = project_history(snapshot, compacted)

    assert projected[0].text == "<summary>\n工作摘要\n</summary>"
    assert projected[0].sender == "context"
    assert projected[1:] == [messages[1], messages[2], messages[4], messages[5]]
    assert messages[3].text == "第一步"
    assert project_history(history_snapshot(messages, initial_count=1), None) == messages


def test_images_follow_protected_input_steer_and_complete_retained_tail(tmp_path: Path) -> None:
    """重复压缩仅移出非保护旧前缀，保留图片消息与尾部完整工具组及其 metadata。"""
    ref = ImageFileRef(path=tmp_path / "plot.png", mime_type="image/png", width=40, height=20)
    images = [
        ImageBlock(original=ref, model=ref, name=name) for name in ("old", "input", "steer", "tail")
    ]
    messages = [
        Msg.user([images[0]]),
        Msg.assistant("old answer"),
        Msg.user("BCI", sender="context", metadata={"context_kind": "before_current_input"}),
        Msg.user([images[1]]),
        Msg.assistant([ToolUseBlock(id="a", name="read"), ToolUseBlock(id="b", name="read")]),
        Msg.tool_result(tool_use_id="a", content=[images[0]]),
        Msg.tool_result(tool_use_id="b", content=[images[0]]),
        Msg.user([images[2]]),
        Msg.assistant(
            [ToolUseBlock(id="c", name="read"), ToolUseBlock(id="d", name="read")],
            metadata={"provider_replay": {"opaque": "unchanged"}},
        ),
        Msg.tool_result(tool_use_id="c", content=[images[3]]),
        Msg.tool_result(tool_use_id="d", content=[images[3]]),
    ]
    before = [message.model_dump_json() for message in messages]
    snapshot = history_snapshot(messages, initial_count=2)
    assert snapshot.protected_indices == (2, 3, 7)
    assert project_history(snapshot, None) == messages
    previous = None
    for covered in (7, 8):
        snapshot = history_snapshot(messages, initial_count=2, compaction=previous)
        current = SessionCompaction(summary="结论", covered_message_count=covered)
        projected = project_history(snapshot, current)
        assert projected[1:] == [messages[2], messages[3], messages[7], *messages[8:]]
        assert projected[2].blocks[0] is images[1]
        assert projected[3].blocks[0] is images[2]
        assert projected[4] is messages[8]
        assert projected[5].tool_results[0].content == [images[3]]
        assert projected[6].tool_results[0].content == [images[3]]
        previous = current
    assert [message.model_dump_json() for message in messages] == before


def test_repeated_projection_uses_only_latest_summary_with_one_wrapper() -> None:
    messages = [Msg.user("任务"), Msg.assistant("第一步"), Msg.assistant("第二步")]
    compacted = SessionCompaction(summary="第一次", covered_message_count=2)
    snapshot = history_snapshot(messages, compaction=compacted)

    first = project_history(snapshot, compacted)
    second = project_history(snapshot, SessionCompaction(summary="第二次", covered_message_count=3))

    assert first[1:] == [messages[0], messages[2]]
    assert [message.text for message in second] == ["<summary>\n第二次\n</summary>", "任务"]


@pytest.mark.parametrize("covered", [0, 2])
@pytest.mark.parametrize("with_images", [False, True])
def test_cuts_within_current_run_and_keeps_entire_parallel_tool_batch(
    covered: int, with_images: bool, tmp_path: Path
) -> None:
    messages = [
        Msg.user("旧" * 500),
        Msg.assistant("旧" * 500),
        Msg.user("当前任务"),
        Msg.assistant(
            [
                TextBlock(text="长" * 600),
                ToolUseBlock(id="a", name="read", input={}),
                ToolUseBlock(id="b", name="read", input={}),
            ]
        ),
        Msg.tool_result(tool_use_id="a", content="A" * 80),
        Msg.tool_result(tool_use_id="b", content="B" * 80, is_error=True),
        Msg.assistant([TextBlock(text="近" * 100), ToolUseBlock(id="c", name="read")]),
        Msg.tool_result(tool_use_id="c", content="C" * 50),
    ]
    if with_images:
        ref = ImageFileRef(path=tmp_path / "plot.png", mime_type="image/png", width=40, height=20)
        image = ImageBlock(original=ref, model=ref)
        messages[2] = Msg.user([TextBlock(text="当前任务"), image])
        for index in (4, 5, 7):
            messages[index].tool_results[0].content.append(image)

    previous = (
        SessionCompaction(summary="已有摘要", covered_message_count=covered) if covered else None
    )
    end = _select(
        messages,
        initial_count=2,
        previous=previous,
        config=CompactionConfig(input_budget_tokens=1200),
    )

    assert end == 6
    projected = project_history(
        history_snapshot(messages, initial_count=2, compaction=previous),
        SessionCompaction(summary="旧步骤摘要", covered_message_count=end),
    )
    assert projected[1:] == [messages[2], messages[6], messages[7]]


def test_older_bci_and_original_input_remain_one_group() -> None:
    messages = [
        Msg.assistant("旧" * 500),
        Msg.user("环境" * 50, sender="context", metadata={"context_kind": "before_current_input"}),
        Msg.user("任务" * 50),
        Msg.assistant("最新回复"),
    ]

    assert _select(messages, initial_count=len(messages)) == 1


def test_oversized_previous_group_does_not_displace_small_latest_group() -> None:
    messages = [
        Msg.user("当前任务"),
        Msg.assistant([TextBlock(text="大" * 120000), ToolUseBlock(id="a", name="read")]),
        Msg.tool_result(tool_use_id="a", content="完成"),
        Msg.assistant("近" * 3000),
    ]

    assert _select(messages, config=CompactionConfig()) == 3


def test_oversized_latest_group_allows_empty_suffix_and_preserves_anchor() -> None:
    messages = [
        Msg.user("当前任务"),
        Msg.assistant([ToolUseBlock(id="a", name="read")]),
        Msg.tool_result(tool_use_id="a", content="大" * 120000),
    ]

    end = _select(messages, config=CompactionConfig())

    assert end == len(messages)
    assert project_history(
        history_snapshot(messages),
        SessionCompaction(summary="结果摘要", covered_message_count=end),
    )[1:] == [messages[0]]


def test_anchor_does_not_consume_recent_target_but_consumes_request_capacity() -> None:
    messages = [
        Msg.user("任务" * 150),
        Msg.assistant("旧" * 600),
        Msg.assistant("近" * 160),
    ]

    assert _select(messages) == 2
    messages[0] = Msg.user("任务" * 350)
    assert _select(messages) == 3


def test_planning_counts_fixed_content_tools_schema_and_summary_wrapper() -> None:
    messages = [Msg.assistant("旧" * 500), Msg.assistant("近" * 160)]

    def with_fixed_content(history: list[Msg]) -> LLMRequest:
        return LLMRequest(
            model="test-model",
            messages=[Msg.system("规则" * 250), *history],
            tools=[{"name": "tool", "input_schema": {}, "description": "工具定义" * 15}],
            response_format={"name": "answer", "schema": {"description": "响应schema" * 10}},
        )

    assert _select(messages, initial_count=2) == 1
    assert _select(messages, initial_count=2, build_request=with_fixed_content) == 2


def test_summary_cap_is_only_a_planning_reservation() -> None:
    messages = [Msg.user("很长输入" * 200), Msg.assistant("旧执行结果")]

    assert _select(messages) == 2


def test_no_new_covered_range_returns_none_without_rewriting_old_summary() -> None:
    messages = [Msg.user("任务"), Msg.assistant("已归档")]
    previous = SessionCompaction(summary="已有摘要", covered_message_count=2)

    assert _select(messages, previous=previous) is None
    assert _select([]) is None


def test_open_tool_batch_is_never_selected_as_covered_prefix() -> None:
    messages = [
        Msg.user("旧" * 900),
        Msg.assistant([ToolUseBlock(id="a", name="read"), ToolUseBlock(id="b", name="read")]),
        Msg.tool_result(tool_use_id="a", content="大" * 900),
    ]

    assert _select(messages, initial_count=len(messages)) == 1


def test_existing_summary_boundary_only_advances_over_new_complete_groups() -> None:
    messages = [Msg.user("旧任务"), Msg.assistant("旧步骤"), Msg.assistant("新" * 400)]
    previous = SessionCompaction(summary="已有摘要", covered_message_count=2)

    assert _select(messages, initial_count=3, previous=previous) == 3


@pytest.mark.parametrize("covered", [2, 3, 5, 6, 7, 8])
def test_sparse_history_candidates_equal_full_history_projection(covered: int) -> None:
    """摘要推进跨过后缀内输入和steer时，它们转为保护前缀且只出现一次。"""
    messages = [
        Msg.user("old question"),
        Msg.assistant("old answer"),
        Msg.assistant("old continuation"),
        Msg.user("BCI", sender="context", metadata={"context_kind": "before_current_input"}),
        Msg.user("current input"),
        Msg.assistant("working"),
        Msg.user("latest steer"),
        Msg.assistant("new observation"),
    ]
    previous = SessionCompaction(summary="previous", covered_message_count=covered)
    snapshot = history_snapshot(messages, initial_count=3, compaction=previous)
    protected = (3, 4, 6)
    assert snapshot.header.message_count == len(messages)
    assert snapshot.raw_tail == tuple(messages[covered:])
    assert snapshot.protected_indices == protected
    assert snapshot.protected_prefix_messages == tuple(
        (index, messages[index]) for index in protected if index < covered
    )
    for end in (index for index in (2, 3, 5, 6, 7, 8) if index >= covered):
        candidate = SessionCompaction(summary=f"summary-{end}", covered_message_count=end)
        expected = [
            Msg.user(f"<summary>\nsummary-{end}\n</summary>", sender="context"),
            *(messages[index] for index in protected if index < end),
            *messages[end:],
        ]
        assert project_history(snapshot, candidate) == expected


def test_empty_tail_keeps_prefix_anchors_without_new_compaction_work() -> None:
    """C=N时只装配摘要和保护原文，选切点不计量任何候选。"""
    messages = [Msg.assistant("old"), Msg.user("input"), Msg.assistant("work"), Msg.user("steer")]
    compaction = SessionCompaction(summary="all covered", covered_message_count=len(messages))
    snapshot = history_snapshot(messages, initial_count=1, compaction=compaction)
    assert snapshot.raw_tail == ()
    assert snapshot.protected_indices == (1, 3)
    assert project_history(snapshot, compaction)[1:] == [messages[1], messages[3]]
    requests: list[list[Msg]] = []

    def build(history: list[Msg]) -> MeasuredRequest:
        requests.append(history)
        return measure_request(_request(history), _estimate)

    assert (
        select_compaction_end(snapshot=snapshot, config=CompactionConfig(), build_request=build)
        is None
    )
    assert requests == []


def test_documented_boundary_example_preserves_absolute_positions() -> None:
    """设计实例中122从后缀转为保护前缀，94/95仍保留且不重复。"""
    messages = [Msg.assistant(f"message {index}") for index in range(136)]
    messages[94] = Msg.user(
        "BCI", sender="context", metadata={"context_kind": "before_current_input"}
    )
    messages[95] = Msg.user("input")
    messages[122] = Msg.user("latest steer")
    previous = SessionCompaction(summary="previous", covered_message_count=100)
    snapshot = history_snapshot(messages, initial_count=94, compaction=previous)
    assert snapshot.protected_indices == (94, 95, 122)
    assert snapshot.protected_prefix_messages == ((94, messages[94]), (95, messages[95]))
    assert snapshot.raw_tail == tuple(messages[100:])
    assert project_history(snapshot, previous)[1:] == [messages[94], messages[95], *messages[100:]]
    candidate = SessionCompaction(summary="next", covered_message_count=128)
    assert project_history(snapshot, candidate)[1:] == [
        messages[94],
        messages[95],
        messages[122],
        *messages[128:],
    ]
