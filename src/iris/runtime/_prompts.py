"""Runtime 文案渲染与 context 错误归属。"""

from typing import Any

from ..exceptions import IrisContextError, IrisTemplateError
from ..prompts import PromptSnapshot, PromptSource


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
