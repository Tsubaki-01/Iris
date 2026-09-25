"""宿主运行态快照是不可变、完整替换的请求材料。"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from iris.context import ContextBuildScope, ContextContribution, ContextSnapshot
from iris.context.source import render_context_snapshot


def test_snapshot_rendering_keeps_order_and_marks_current_state() -> None:
    snapshot = ContextSnapshot(
        (ContextContribution("document", "报告 A"), ContextContribution("selection", "第二段"))
    )
    message = render_context_snapshot(snapshot)
    assert message.sender == "context"
    assert message.metadata == {"context_kind": "runtime_snapshot"}
    assert message.text.index("document") < message.text.index("selection")
    assert "报告 A" in message.text and "第二段" in message.text
    assert "当前" in message.text and "历史" in message.text
    assert "当前无已提供的运行态条目" in render_context_snapshot(ContextSnapshot()).text


def test_source_values_are_frozen_and_required_by_default() -> None:
    contribution = ContextContribution("document", "报告 A")
    assert contribution.required and contribution.priority == 100
    scope = ContextBuildScope("s", "r", 0, Path.cwd(), "检查报告")
    with pytest.raises(FrozenInstanceError):
        scope.run_input = "另一个任务"
