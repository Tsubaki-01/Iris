"""Runtime 文案渲染与 context 错误归属。"""

from typing import Any

from ..exceptions import IrisContextError, IrisTemplateError
from ..prompts import PromptSnapshot, PromptSource


def compaction_input_variables(
    previous_summary: str | None, serialized_history: str
) -> dict[str, Any]:
    """实际摘要输入和代表输入共用变量投影。"""
    return {
        "previous_summary_or_none": "(none)" if previous_summary is None else previous_summary,
        "serialized_history": serialized_history,
    }


def compaction_prompt_descriptions() -> dict[str, tuple[str, dict[str, Any]]]:
    """说明自然语言摘要与真实模板变量，不创建结构化输出协议。"""
    return {
        "compaction": ("无模板变量。输出非空自然语言摘要，供后续上下文继续使用；不要求 JSON。", {}),
        "compaction_input": (
            "previous_summary_or_none 是旧摘要或 (none)，serialized_history 是本批已提交历史。"
            "输出仍是自然语言摘要，不改变程序的选材与分批规则。",
            compaction_input_variables(None, "[message:0] user: 示例任务"),
        ),
    }


def snapshot_prompts(source: PromptSource) -> PromptSnapshot:
    """取得操作快照，将读取失败归属 runtime 的 context 边界。"""
    try:
        return source.snapshot()
    except IrisTemplateError as exc:
        raise IrisContextError(exc.message, **exc.context) from exc


def render_prompt(snapshot: PromptSnapshot, prompt_id: str, context: dict[str, Any]) -> str:
    """渲染 runtime 文案并保留模板错误的 context 来源。"""
    try:
        return snapshot.render(prompt_id, context)
    except IrisTemplateError as exc:
        raise IrisContextError(exc.message, **exc.context) from exc
