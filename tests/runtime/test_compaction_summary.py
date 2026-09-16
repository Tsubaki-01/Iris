"""滚动摘要输入、分片和响应消费的契约测试。"""

from __future__ import annotations

import pytest

from iris.agents.config.compaction import CompactionConfig
from iris.exceptions import IrisContextCompactionError
from iris.message import LLMRequest, LLMResponse, Msg, TextBlock, ToolResultBlock, ToolUseBlock
from iris.runtime._compaction_summary import (
    consume_summary_response,
    next_summary_batch,
    serialize_history,
)


def _estimate(request: LLMRequest) -> int:
    return sum(len(message.text) for message in request.messages)


def _main_request() -> LLMRequest:
    return LLMRequest(
        model="openai/model",
        messages=[Msg.system("业务系统指令"), Msg.user("尚未归档的当前输入")],
        temperature=0.3,
        top_p=0.8,
        max_tokens=12000,
        tools=[{"type": "function", "function": {"name": "shell"}}],
        tool_choice="required",
        response_format={"type": "json_object"},
        stream=True,
        timeout=10,
        provider_options={"reasoning_effort": "low", "num_retries": 5},
    )


def test_history_preserves_all_blocks_and_necessary_tool_semantics() -> None:
    messages = [
        Msg.user("昨天修复 src/app.py", sender="用户", timestamp=1234567890),
        Msg.assistant(
            [
                TextBlock(text="先检查日志"),
                ToolUseBlock(id="call-9", name="read_file", input={"path": "日志.txt"}),
            ],
            metadata={"usage": {"input_tokens": 987654}, "trace_id": "trace-secret"},
        ),
        Msg.tool_result(
            tool_use_id="call-9",
            name="read_file",
            content="文件无法读取",
            is_error=True,
            metadata={
                "error": {"code": "NOT_FOUND", "message": "缺少日志.txt"},
                "artifact": {"path": "tmp/report.txt"},
                "stats": {"elapsed": 999999},
                "trace_id": "result-trace",
            },
        ),
    ]

    records = serialize_history(messages, start_index=41)
    rendered = "\n".join(record.header + "\n" + record.text for record in records)

    assert len(records) == 4
    assert "message=41" in records[0].header
    assert "role=user" in records[0].header
    assert 'sender="用户"' in records[0].header
    assert "昨天修复 src/app.py" in rendered
    assert "先检查日志" in rendered
    assert '"path":"日志.txt"' in rendered
    assert "call-9" in records[2].header
    assert "call-9" in records[3].header
    assert "read_file" in rendered
    assert "execution_status=failed" in records[3].header
    assert "is_error=true" in records[3].header
    assert "NOT_FOUND" in rendered and "tmp/report.txt" in rendered
    assert "1234567890" not in rendered
    assert "987654" not in rendered and "999999" not in rendered
    assert "trace-secret" not in rendered and "result-trace" not in rendered


def test_summary_request_uses_canonical_seven_headings_and_final_request_options() -> None:
    main = _main_request()
    config = CompactionConfig()
    records = serialize_history([Msg.user("已保存的任务")], start_index=0)

    batch = next_summary_batch(main, None, records, (0, 0), config, _estimate)
    request = batch.request

    assert request.model == main.model
    assert request.temperature == 0.3 and request.top_p == 0.8
    assert request.timeout == 10
    assert request.max_tokens == config.summary_tokens
    assert request.stream is False
    assert request.tools == [] and request.tool_choice is None
    assert request.response_format is None
    assert request.provider_options == {"reasoning_effort": "low", "num_retries": 0}
    assert main.provider_options["num_retries"] == 5
    assert main.stream and main.tools
    assert len(request.messages) == 2
    system, user = (message.text for message in request.messages)
    assert [line for line in system.splitlines() if line.startswith("## ")] == [
        "## Goal & Constraints",
        "## Completed Work & Evidence",
        "## Current State & Unfinished Items",
        "## Decisions & Rationale",
        "## Failed Attempts & Blockers",
        "## Next Steps",
        "## Critical Paths & Identifiers",
    ]
    assert "=== BEGIN PREVIOUS SUMMARY ===\n(none)\n" in user
    assert "已保存的任务" in user
    assert "业务系统指令" not in user and "尚未归档的当前输入" not in user
    assert "current time are not part" in user
    assert batch.next_position == (len(records), 0)


def test_long_completed_result_is_covered_once_in_order_with_call_identity() -> None:
    content = "".join(f"数据行{i:05d}。" for i in range(3000))
    records = serialize_history(
        [Msg.tool_result(tool_use_id="long-call", name="read_file", content=content)],
        start_index=9,
    )
    config = CompactionConfig(input_budget_tokens=6500)
    position = (0, 0)
    previous = None
    processed = ""
    count = 0

    while position[0] < len(records):
        batch = next_summary_batch(_main_request(), previous, records, position, config, _estimate)
        user = batch.request.messages[1].text
        end = len(records[0].text) if batch.next_position[0] else batch.next_position[1]
        expected_fragment = records[0].text[position[1] : end]
        assert expected_fragment in user
        assert "long-call" in user and "message=9" in user and "block=0" in user
        assert "execution_status=completed" in user and "is_error=false" in user
        assert f"[{position[1]},{end})/{len(records[0].text)}" in user
        assert _estimate(batch.request) <= config.input_budget_tokens
        assert batch.next_position != position
        processed += expected_fragment
        previous = f"已处理 {end} 个字符；调用 long-call 已完成，文字覆盖到 {end}。"
        position = batch.next_position
        count += 1

    assert count > 1
    assert processed == records[0].text
    assert content in processed


def test_each_batch_recounts_the_current_larger_working_summary() -> None:
    records = serialize_history([Msg.user("A" * 20000)], start_index=0)
    config = CompactionConfig(input_budget_tokens=6500)
    main = _main_request()
    first = next_summary_batch(main, None, records, (0, 0), config, _estimate)
    larger_summary = "仍有效的旧约束。" * 60
    second = next_summary_batch(
        main, larger_summary, records, first.next_position, config, _estimate
    )

    first_length = first.next_position[1]
    second_length = second.next_position[1] - first.next_position[1]
    assert 0 < second_length < first_length
    assert larger_summary in second.request.messages[1].text
    assert _estimate(second.request) <= config.input_budget_tokens


def test_records_are_merged_in_order_including_empty_content() -> None:
    records = serialize_history(
        [Msg.user(""), Msg.assistant([]), Msg.user("第三条"), Msg.assistant("第四条")],
        start_index=11,
    )
    batch = next_summary_batch(
        _main_request(), "有效的旧摘要", records, (0, 0), CompactionConfig(), _estimate
    )
    user = batch.request.messages[1].text

    assert batch.next_position == (len(records), 0)
    assert user.index("message=11") < user.index("message=12") < user.index("第三条")
    assert user.index("第三条") < user.index("第四条")
    assert "有效的旧摘要" in user


def test_minimum_fragment_failure_does_not_drop_the_previous_summary() -> None:
    records = serialize_history([Msg.user("新记录")], start_index=0)

    with pytest.raises(IrisContextCompactionError) as error:
        next_summary_batch(
            _main_request(),
            "必须保留的旧约束" * 1000,
            records,
            (0, 0),
            CompactionConfig(input_budget_tokens=6500),
            _estimate,
        )

    assert error.value.runtime_code == "CONTEXT_COMPACTION_UNAVAILABLE"


def test_summary_consumption_accepts_text_without_parsing_its_headings() -> None:
    response = LLMResponse(
        provider="test",
        finish_reason="stop",
        content=[TextBlock(text="  普通摘要"), TextBlock(text="继续工作。  ")],
    )

    assert consume_summary_response(response) == "普通摘要\n继续工作。"


@pytest.mark.parametrize("finish_reason", ["", "length", "tool_calls", "content_filter"])
def test_summary_consumption_rejects_incomplete_termination(finish_reason: str) -> None:
    with pytest.raises(IrisContextCompactionError) as error:
        consume_summary_response(
            LLMResponse(
                provider="test", finish_reason=finish_reason, content=[TextBlock(text="部分摘要")]
            )
        )

    assert error.value.runtime_code == "CONTEXT_COMPACTION_FAILED"


@pytest.mark.parametrize(
    "content",
    [
        [],
        [TextBlock(text=" \n ")],
        [TextBlock(text="摘要"), ToolUseBlock(id="call", name="shell")],
        [ToolResultBlock(tool_use_id="call", content="工具结果")],
    ],
)
def test_summary_consumption_rejects_empty_or_nontext_content(
    content: list[TextBlock | ToolUseBlock | ToolResultBlock],
) -> None:
    with pytest.raises(IrisContextCompactionError) as error:
        consume_summary_response(
            LLMResponse(provider="test", finish_reason="stop", content=content)
        )

    assert error.value.runtime_code == "CONTEXT_COMPACTION_FAILED"
