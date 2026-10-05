"""记忆提示来源、固定响应契约与领域错误转换。"""

import json
from typing import Any

from pydantic import BaseModel

from ..exceptions import IrisMemoryError, IrisTemplateError
from ..prompts import PromptSnapshot, PromptSource


def snapshot_memory_prompts(source: PromptSource | None) -> PromptSnapshot:
    """在完整操作入口固定显式来源，不推断工作目录。"""
    if source is None:
        raise IrisMemoryError("memory prompt 来源未配置")
    try:
        return source.snapshot()
    except IrisTemplateError as exc:
        raise IrisMemoryError("memory prompt 来源读取失败", **exc.context) from exc


def render_memory_prompt(snapshot: PromptSnapshot, prompt_id: str, context: dict[str, Any]) -> str:
    """渲染记忆阶段指令，并让已有生成失败记录接收记忆领域错误。"""
    try:
        return snapshot.render(prompt_id, context)
    except IrisTemplateError as exc:
        raise IrisMemoryError("memory 生成模板渲染失败", **exc.context) from exc


def structured_memory_prompt(
    snapshot: PromptSnapshot, prompt_id: str, instructions: str, response_model: type[BaseModel]
) -> str:
    """在可编辑策略后追加由领域拥有的固定说明与真实响应 schema。"""
    strategy = render_memory_prompt(snapshot, prompt_id, {})
    return f"{strategy}\n\n固定输出契约：\n{memory_output_contract(instructions, response_model)}"


def memory_output_contract(instructions: str, response_model: type[BaseModel]) -> str:
    """实际请求和领域说明共用同一份输出协议。"""
    schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
    return f"{instructions}\n只返回符合以下 JSON Schema 的 JSON，不附加文字或代码围栏。\n{schema}"
