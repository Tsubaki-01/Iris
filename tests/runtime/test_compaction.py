"""历史压缩的原文保护、投影与合法切点测试。"""

from __future__ import annotations

import json
from collections.abc import Callable

from iris.agents import CompactionConfig
from iris.lifecycle import SessionCompaction
from iris.message import LLMRequest, Msg, TextBlock, ToolResultBlock, ToolUseBlock
from iris.runtime.compaction import (
    project_history,
    protected_message_indices,
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
                count += len(block.tool_use_id) + len(block.content)
    count += len(json.dumps(request.tools)) if request.tools else 0
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
        messages=messages,
        previous_compaction=previous,
        protected_indices=protected_message_indices(messages, initial_count),
        config=config or CompactionConfig(input_budget_tokens=1000),
        build_request=build_request,
        estimate_input_tokens=_estimate,
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

    assert protected_message_indices(messages, 3) == (3, 4, 7)


def test_current_run_without_bci_never_borrows_old_context() -> None:
    messages = [
        Msg.user("旧环境", sender="context", metadata={"context_kind": "before_current_input"}),
        Msg.user("旧任务"),
        Msg.assistant("完成"),
        Msg.user("新任务"),
        Msg.assistant("处理中"),
    ]

    assert protected_message_indices(messages, 3) == (3,)
    assert protected_message_indices(messages, len(messages)) == ()


def test_dynamic_memories_can_be_compressed_without_losing_bci_or_user() -> None:
    """逐条动态快照不占用本轮输入锚点，也不能被误识别为 steer。"""
    messages = [
        Msg.user("记忆A", sender="context", metadata={"context_kind": "memory", "item_id": "a"}),
        Msg.user("记忆B", sender="context", metadata={"context_kind": "memory", "item_id": "b"}),
        Msg.user("本轮环境", sender="context", metadata={"context_kind": "before_current_input"}),
        Msg.user("原始问题"),
        Msg.assistant("处理中"),
        Msg.user("补充要求"),
    ]

    protected = protected_message_indices(messages, 0)

    assert protected == (2, 3, 5)
    projected = project_history(
        messages, SessionCompaction(summary="历史摘要", covered_message_count=5), protected
    )
    assert projected[1:] == [messages[2], messages[3], messages[5]]


def test_dynamic_memory_before_user_is_a_valid_compaction_cut() -> None:
    """没有BCI时，可以仅压缩动态记忆而保留原始问题。"""
    messages = [
        Msg.user("记忆" * 600, sender="context", metadata={"context_kind": "memory"}),
        Msg.user("当前问题"),
    ]

    assert protected_message_indices(messages, 0) == (1,)
    assert _select(messages, config=CompactionConfig(input_budget_tokens=1000)) == 1


def test_projection_restores_covered_anchors_once_in_original_order() -> None:
    messages = [
        Msg.user("旧历史"),
        Msg.user("当前环境", sender="context", metadata={"context_kind": "before_current_input"}),
        Msg.user("当前任务"),
        Msg.assistant("第一步"),
        Msg.user("最新方向"),
        Msg.assistant("第二步"),
    ]
    protected = protected_message_indices(messages, 1)
    compacted = SessionCompaction(summary="工作摘要", covered_message_count=4)

    projected = project_history(messages, compacted, protected)

    assert projected[0].text == "<summary>\n工作摘要\n</summary>"
    assert projected[0].sender == "context"
    assert projected[1:] == [messages[1], messages[2], messages[4], messages[5]]
    assert messages[3].text == "第一步"
    assert project_history(messages, None, protected) == messages


def test_repeated_projection_uses_only_latest_summary_with_one_wrapper() -> None:
    messages = [Msg.user("任务"), Msg.assistant("第一步"), Msg.assistant("第二步")]
    protected = protected_message_indices(messages, 0)

    first = project_history(
        messages, SessionCompaction(summary="第一次", covered_message_count=2), protected
    )
    second = project_history(
        messages, SessionCompaction(summary="第二次", covered_message_count=3), protected
    )

    assert first[1:] == [messages[0], messages[2]]
    assert [message.text for message in second] == ["<summary>\n第二次\n</summary>", "任务"]


def test_cuts_within_current_run_and_keeps_entire_parallel_tool_batch() -> None:
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

    end = _select(messages, initial_count=2, config=CompactionConfig(input_budget_tokens=1200))

    assert end == 6
    projected = project_history(
        messages,
        SessionCompaction(summary="旧步骤摘要", covered_message_count=end),
        protected_message_indices(messages, 2),
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
        messages,
        SessionCompaction(summary="结果摘要", covered_message_count=end),
        (0,),
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
            tools=[{"description": "工具定义" * 15}],
            response_format={"description": "响应schema" * 10},
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
