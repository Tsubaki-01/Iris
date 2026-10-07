"""真实上下文准备过程的不可变观察事实，不参与恢复和执行裁决。"""

from dataclasses import dataclass
from typing import Literal

from ..lifecycle import RunErrorInfo


@dataclass(frozen=True, slots=True)
class ContextDecision:
    """真实选择分支对一个来源作出的决定。"""

    subject_kind: str
    subject_ref: str
    action: Literal["retained", "replaced", "removed", "summarized", "unchanged"]
    reason_code: str
    related_ref: str | None = None


@dataclass(frozen=True, slots=True)
class ContextStage:
    """完整请求的阶段计量；candidate 不表示已经应用。"""

    index: int
    kind: str
    outcome: Literal["applied", "skipped", "candidate", "rejected"]
    before_input_tokens: int | None = None
    after_input_tokens: int | None = None
    decisions: tuple[ContextDecision, ...] = ()
    compaction_ref: str | None = None


@dataclass(frozen=True, slots=True)
class ContextPreparation:
    """一次 before_model 准备的完整只读结果。"""

    preparation_id: str
    configuration_snapshot_id: str
    session_id: str
    run_id: str
    activation_id: str
    step_index: int
    phase: Literal["preparing", "ready", "failed", "cancelled"]
    input_budget_tokens: int
    trigger_tokens: int
    stages: tuple[ContextStage, ...] = ()
    selected_tool_names: tuple[str, ...] = ()
    selected_contribution_keys: tuple[str, ...] = ()
    protected_refs: tuple[str, ...] = ()
    final_input_tokens: int | None = None
    final_request_ref: str | None = None
    error: RunErrorInfo | None = None


__all__ = ["ContextDecision", "ContextStage", "ContextPreparation"]
