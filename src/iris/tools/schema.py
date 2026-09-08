"""工具 schema 生成与 provider 包装函数。

负责将 Python 代码（Pydantic 模型或普通函数签名）转换为各个 LLM Provider
（如 OpenAI, Anthropic）所需的不同格式的工具定义 Schema。

Example:
    schema = schema_from_pydantic_model(MyModel)
"""

# region imports
from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import BaseModel, Field, create_model

from ..exceptions import IrisToolValidationError

# endregion


@dataclass(slots=True)
class DocstringInfo:
    """Google Style docstring 提取结果。"""

    summary: str = ""
    args: dict[str, str] = field(default_factory=dict)
    returns: str = ""
    examples: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DocstringSchemaExtractor:
    """从函数 docstring 提取 schema 可用的说明文本。"""

    def extract(self, func: Callable[..., Any]) -> DocstringInfo:
        """提取函数 docstring 中的概要、参数说明和示例。

        Args:
            func (Callable[..., Any]): 需要分析的函数。

        Returns:
            DocstringInfo: 可用于合成工具 schema 的说明信息。
        """
        doc = inspect.getdoc(func) or ""
        if not doc:
            return DocstringInfo(warnings=["缺少 docstring"])
        lines = doc.splitlines()
        summary = lines[0].strip() if lines else ""
        info = DocstringInfo(summary=summary)
        section = ""
        current_arg = ""
        for raw_line in lines[1:]:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped in {"Args:", "Arguments:", "Parameters:"}:
                section = "args"
                current_arg = ""
                continue
            if stripped in {"Returns:", "Raises:", "Example:", "Examples:"}:
                section = stripped.rstrip(":").lower()
                current_arg = ""
                continue
            if section == "args":
                if ":" in stripped:
                    arg_name, description = stripped.split(":", 1)
                    current_arg = arg_name.split(" ", 1)[0].strip()
                    info.args[current_arg] = description.strip()
                elif current_arg:
                    info.args[current_arg] = f"{info.args[current_arg]} {stripped}".strip()
                continue
            if section == "returns":
                info.returns = f"{info.returns} {stripped}".strip()
            elif section in {"example", "examples"}:
                info.examples.append(stripped)
        return info


def schema_from_pydantic_model(model: type[BaseModel]) -> dict[str, Any]:
    """从输入校验模型导出完整 JSON Schema。"""
    return model.model_json_schema()


def schema_from_callable(
    func: Callable[..., Any],
    *,
    preset_kwargs: set[str],
) -> dict[str, Any]:
    """从函数的动态输入模型导出包含参数说明的 JSON Schema。"""
    return schema_from_pydantic_model(callable_input_model(func, preset_kwargs))


def callable_input_model(
    func: Callable[..., Any],
    preset_kwargs: set[str],
) -> type[BaseModel]:
    """从函数签名、类型注解和参数说明构造唯一输入契约。"""
    fields: dict[str, Any] = {}
    type_hints = _type_hints(func)
    doc_info = DocstringSchemaExtractor().extract(func)
    for name, parameter in inspect.signature(func).parameters.items():
        if name in preset_kwargs:
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            raise IrisToolValidationError("工具函数只支持普通参数和关键字参数", parameter=name)
        annotation = type_hints.get(name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            raise IrisToolValidationError("工具函数参数必须包含类型注解", parameter=name)
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[name] = (annotation, Field(default=default, description=doc_info.args.get(name)))
    return create_model(f"{func.__name__.title().replace('_', '')}ToolInput", **fields)  # ty:ignore[unresolved-attribute]


def to_openai_chat_tool_schema(definition: Any) -> dict[str, Any]:
    """生成 OpenAI Chat Completions 工具 schema。"""
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": definition.input_schema,
        },
    }


def to_openai_responses_tool_schema(definition: Any) -> dict[str, Any]:
    """生成 OpenAI Responses 工具 schema。"""
    return {
        "type": "function",
        "name": definition.name,
        "description": definition.description,
        "parameters": definition.input_schema,
        "strict": False,
    }


def to_anthropic_tool_schema(definition: Any) -> dict[str, Any]:
    """生成 Anthropic Messages 工具 schema。"""
    return {
        "name": definition.name,
        "description": definition.description,
        "input_schema": definition.input_schema,
    }


def _type_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """解析 postponed annotations。"""
    try:
        return get_type_hints(func, include_extras=True)
    except NameError as exc:
        raise IrisToolValidationError("工具函数类型注解无法解析", error=str(exc)) from exc
