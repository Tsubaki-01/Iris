"""上下文计量复用与有界候选探测，不改变完整预算和原文覆盖。"""

from __future__ import annotations

import re

from iris.agents import CompactionConfig
from iris.lifecycle import SessionContextWindow
from iris.message import LLMRequest, Msg
from iris.runtime._compaction_summary import next_summary_batch, serialize_history
from iris.runtime.compaction import select_compaction_end
from iris.runtime.memory_context import select_context_window
from iris.utils import TemplateRenderer

from .test_compaction_summary import _DEFAULT_PROMPT, _estimate, _main_request
from .test_memory_context import TextTokenProvider, _request


def test_many_small_records_need_few_complete_request_estimates() -> None:
    """整批能装下时，无需对每一个逐渐增长的前缀重新计量。"""
    records = serialize_history([Msg.user(f"record {i}: Atlas uses uv") for i in range(256)], 0)
    calls = 0

    def estimate(request: LLMRequest) -> int:
        nonlocal calls
        calls += 1
        return _estimate(request)

    config = CompactionConfig()
    batch = next_summary_batch(
        _main_request(),
        None,
        records,
        (0, 0),
        config,
        estimate,
        system_prompt=_DEFAULT_PROMPT,
        prompt_renderer=TemplateRenderer(),
    )
    assert batch.next_position == (256, 0)
    assert calls < 20
    assert _estimate(batch.request) <= config.input_budget_tokens
    text = batch.request.messages[1].text
    assert text.count("[record ") == 256
    assert text.index("record 0:") < text.index("record 128:") < text.index("record 255:")


def test_closed_history_does_not_recount_the_same_empty_suffix() -> None:
    """最靠后完整切点就是历史末尾时，直接复用空后缀的完整估算。"""
    requests: list[LLMRequest] = []

    def estimate(request: LLMRequest) -> int:
        requests.append(request)
        return _estimate(request)

    end = select_compaction_end(
        messages=[Msg.user("old history")],
        previous_compaction=None,
        protected_indices=(),
        config=CompactionConfig(input_budget_tokens=1000),
        build_request=lambda history: LLMRequest(model="test", messages=history),
        estimate_input_tokens=estimate,
    )
    assert end == 1
    assert len(requests) == 1


def test_identical_overview_candidates_reuse_full_request_and_token_count() -> None:
    """尚无概览的同一候选，即使总输入超额也不重复构造和计量。"""
    navigation = SessionContextWindow(memory_overview="no published facts", mode="navigation")
    built: list[SessionContextWindow] = []
    counted: list[LLMRequest] = []

    class Provider(TextTokenProvider):
        """记录完整请求的估算，保留现有测试计量定义。"""

        def estimate_input_tokens(self, request: LLMRequest) -> int:
            counted.append(request)
            return super().estimate_input_tokens(request)

    def build(window: SessionContextWindow) -> LLMRequest:
        built.append(window)
        return _request(window)

    selected = select_context_window(
        candidates=(navigation, navigation),
        build_request=build,
        provider=Provider(),
        memory_budget_tokens=100,
        input_budget_tokens=1,
    )
    assert len(counted) == len(built) == 2
    window, request, tokens = selected
    assert window is navigation and request is counted[-1]
    assert tokens == TextTokenProvider().estimate_input_tokens(request)


def test_nonzero_offset_and_empty_records_preserve_sequential_coverage() -> None:
    """跨批次起点只裁剪第一条正文，空正文仍保留身份并推进位置。"""
    records = serialize_history(
        [Msg.user("skip"), Msg.user("prefix-tail"), Msg.user(""), Msg.user("last")], 0
    )
    batch = next_summary_batch(
        _main_request(),
        "previous",
        records,
        (1, 7),
        CompactionConfig(),
        _estimate,
        system_prompt=_DEFAULT_PROMPT,
        prompt_renderer=TemplateRenderer(),
    )
    text = batch.request.messages[1].text
    assert batch.next_position == (4, 0)
    assert "prefix-" not in text and "tail" in text
    assert "message=0" not in text
    assert text.index("message=1") < text.index("message=2") < text.index("message=3")
    assert "text_coverage=partial [7,11)/11" in text
    assert "text_coverage=complete [0,0)/0" in text


def test_nonmonotonic_estimates_return_only_an_accepted_request() -> None:
    """候选 token 不严格单调时仍返回实际合预算的请求，不要求最大装填率。"""
    records = serialize_history([Msg.user(f"record {i}") for i in range(9)], 0)
    accepted: list[LLMRequest] = []

    def estimate(request: LLMRequest) -> int:
        count = request.messages[1].text.count("[record ")
        tokens = 10000 if count in {2, 5, 8, 9} else 100
        if tokens <= 6500:
            accepted.append(request)
        return tokens

    batch = next_summary_batch(
        _main_request(),
        None,
        records,
        (0, 0),
        CompactionConfig(input_budget_tokens=6500),
        estimate,
        system_prompt=_DEFAULT_PROMPT,
        prompt_renderer=TemplateRenderer(),
    )
    assert any(batch.request is request for request in accepted)
    assert batch.next_position > (0, 0)
    assert estimate(batch.request) <= 6500


def test_many_records_across_batches_are_covered_once_without_gaps() -> None:
    """多个完整前缀与末条分片交替时，实际请求覆盖恰好等于持久原文。"""
    records = serialize_history([Msg.user(f"source-{i}:" * 450) for i in range(19)], 0)
    config = CompactionConfig(input_budget_tokens=6500)
    position = (0, 0)
    consumed = [0] * len(records)
    batches = 0
    while position[0] < len(records):
        batch = next_summary_batch(
            _main_request(),
            "current summary",
            records,
            position,
            config,
            _estimate,
            system_prompt=_DEFAULT_PROMPT,
            prompt_renderer=TemplateRenderer(),
        )
        assert batch.next_position > position
        assert _estimate(batch.request) <= config.input_budget_tokens
        text = batch.request.messages[1].text
        for match in re.finditer(
            r"\[record message=(\d+).*?text_coverage=\w+ \[(\d+),(\d+)\)/\d+\]\n", text
        ):
            index, start, end = map(int, match.groups())
            assert start == consumed[index]
            assert text[match.end() : match.end() + end - start] == records[index].text[start:end]
            consumed[index] = end
        position = batch.next_position
        batches += 1
    assert batches > 1
    assert consumed == [len(record.text) for record in records]


def test_no_room_for_next_record_keeps_last_successful_request() -> None:
    """下一条连 header 都放不下时，请求仍对应已确认的完整前缀。"""
    records = serialize_history([Msg.user("first"), Msg.user("next" * 100)], 0)
    reference = next_summary_batch(
        _main_request(),
        None,
        records[:1],
        (0, 0),
        CompactionConfig(),
        _estimate,
        system_prompt=_DEFAULT_PROMPT,
        prompt_renderer=TemplateRenderer(),
    )
    batch = next_summary_batch(
        _main_request(),
        None,
        records,
        (0, 0),
        CompactionConfig(input_budget_tokens=_estimate(reference.request)),
        _estimate,
        system_prompt=_DEFAULT_PROMPT,
        prompt_renderer=TemplateRenderer(),
    )
    assert batch.next_position == (1, 0)
    assert batch.request.messages[1].text == reference.request.messages[1].text
