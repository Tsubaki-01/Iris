"""诊断必须复用真实完整请求计量，不改变选择与请求。"""

from iris.agents import ContextPolicyConfig
from iris.message import LLMRequest
from iris.runtime._context_diagnostics import ContextPreparationRecorder
from iris.runtime._context_projection import project_context_request
from iris.runtime._request_measurement import measure_request
from iris.runtime.diagnostics import ContextPreparation

from .test_context_projection import _batch, _estimate


def test_diagnostics_reuse_existing_measurements_and_record_real_decisions() -> None:
    """开启记录不多调用 estimator，重复折叠与近期保护指向真实位置。"""
    raw = [*_batch("body " * 200), *_batch("body " * 200), *_batch("body " * 200)]
    request = LLMRequest(
        model="test", messages=raw, tools=[{"name": "context_read", "input_schema": {}}]
    )
    counts: list[int] = []

    def estimate(value: LLMRequest) -> int:
        tokens = _estimate(value)
        counts.append(tokens)
        return tokens

    kwargs = dict(
        source_indices={id(message): i for i, message in enumerate(raw)},
        config=ContextPolicyConfig(preserve_recent_tool_groups=2),
        trigger_tokens=100,
        estimate_input_tokens=estimate,
    )
    expected, _ = project_context_request(measure_request(request, estimate), **kwargs)
    original_counts = list(counts)
    counts.clear()
    facts: list[ContextPreparation] = []
    diagnostic = ContextPreparationRecorder(
        configuration_snapshot_id="config",
        session_id="s",
        run_id="r",
        activation_id="a",
        step_index=0,
        input_budget_tokens=1000,
        trigger_tokens=100,
        publish=facts.append,
    )
    with diagnostic:
        actual, _ = project_context_request(
            measure_request(request, estimate), **kwargs, diagnostics=diagnostic
        )
        diagnostic.ready(actual, ())
    assert actual == expected
    assert counts == original_counts
    assert facts[0].final_input_tokens == actual.input_tokens
    decisions = [decision for stage in facts[0].stages for decision in stage.decisions]
    assert any(
        decision.reason_code == "duplicate_observation" and decision.action == "replaced"
        for decision in decisions
    )
    assert any(decision.reason_code == "recent_group_protected" for decision in decisions)
    assert all(
        stage.after_input_tokens in counts
        for stage in facts[0].stages
        if stage.after_input_tokens is not None
    )
