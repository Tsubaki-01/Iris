"""记忆生成模板的资源定位与领域错误转换。"""

from pathlib import Path
from typing import Any

from ..exceptions import IrisMemoryError, IrisTemplateError
from ..utils import TemplateRenderer

_PROMPT_DIRECTORY = Path(__file__).resolve().parents[1] / "prompts"


def render_memory_prompt(renderer: TemplateRenderer, filename: str, context: dict[str, Any]) -> str:
    """渲染记忆阶段指令，并让已有生成失败记录接收记忆领域错误。"""
    try:
        return renderer.render_file(_PROMPT_DIRECTORY / filename, context)
    except IrisTemplateError as exc:
        raise IrisMemoryError("memory 生成模板渲染失败", **exc.context) from exc
