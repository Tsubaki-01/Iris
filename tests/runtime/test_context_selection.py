"""完整请求按重复折叠、可选材料、旧观察的唯一顺序减载。"""

from iris.agents import ContextPolicyConfig
from iris.context import ContextContribution, ContextSnapshot
from iris.message import LLMRequest, Msg
from iris.runtime._context_projection import project_context_request
from tests.runtime.test_context_projection import _batch, _estimate


def _project(
    history: list[Msg],
    snapshot: ContextSnapshot,
    *,
    trigger: int,
    select: bool = True,
    tools: bool = True,
) -> tuple[LLMRequest, ContextSnapshot | None]:
    return project_context_request(
        LLMRequest(
            model="test",
            messages=[Msg.system("required-system"), *history],
            tools=[{"function": {"name": "context_read"}}] if tools else [],
        ),
        source_indices={id(message): index for index, message in enumerate(history)},
        config=ContextPolicyConfig(preserve_recent_tool_groups=0, old_result_preview_chars=0),
        trigger_tokens=trigger,
        estimate_input_tokens=_estimate,
        snapshot=snapshot,
        select_optional=select,
    )


def test_duplicate_folding_precedes_optional_removal() -> None:
    body = "same" * 1000
    snapshot = ContextSnapshot((ContextContribution("optional", "notes" * 100, required=False),))
    request, selected = _project([*_batch(body), *_batch(body)], snapshot, trigger=5500)
    assert selected is snapshot
    assert "重复正文见" in request.messages[2].tool_results[0].content
    assert "notes" in request.messages[-1].text


def test_optional_removal_precedes_old_observation_shortening() -> None:
    body = "unique" * 500
    snapshot = ContextSnapshot((ContextContribution("optional", "notes" * 1000, required=False),))
    request, selected = _project(_batch(body), snapshot, trigger=4000)
    assert selected.contributions == ()
    assert request.messages[2].tool_results[0].content == body


def test_priority_tie_required_and_frozen_selection_without_tools() -> None:
    required = ContextContribution("required", "required-value", priority=-100)
    first = ContextContribution("first", "A" * 300, required=False, priority=10)
    second = ContextContribution("second", "B" * 300, required=False, priority=10)
    lowest = ContextContribution("lowest", "C" * 300, required=False, priority=1)
    snapshot = ContextSnapshot((required, first, second, lowest))
    request, selected = _project([Msg.user("user")], snapshot, trigger=600, tools=False)
    assert selected.contributions == (required, first)
    assert "required-value" in request.messages[-1].text
    final, frozen = _project([], selected, trigger=5000, select=False, tools=False)
    assert frozen is selected
    assert "[second]" not in final.messages[-1].text and "[lowest]" not in final.messages[-1].text
    _, next_step = _project([], snapshot, trigger=5000, tools=False)
    assert next_step is snapshot
