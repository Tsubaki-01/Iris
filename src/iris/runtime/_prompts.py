"""Runtime 文案渲染与 context 错误归属。"""

from pathlib import Path
from typing import Any

from ..exceptions import IrisContextError, IrisTemplateError
from ..utils import TemplateRenderer


def render_prompt(renderer: TemplateRenderer, template_path: Path, context: dict[str, Any]) -> str:
    """渲染 runtime 文案并保留模板错误的 context 来源。"""
    try:
        return renderer.render_file(template_path, context)
    except IrisTemplateError as exc:
        raise IrisContextError(exc.message, **exc.context) from exc
