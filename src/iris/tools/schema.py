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
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field, create_model

from ..exceptions import IrisToolValidationError

# endregion


def schema_from_pydantic_model(model: type[BaseModel]) -> dict[str, Any]:
    """从 Pydantic 模型生成最小 JSON Schema object。"""
    schema = model.model_json_schema()
    result: dict[str, Any] = {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }
    if "$defs" in schema:
        result["$defs"] = schema["$defs"]
    return result


def schema_from_callable(
    func: Callable[..., Any],
    *,
    preset_kwargs: set[str],
) -> dict[str, Any]:
    """从函数签名生成阶段 1 支持的 JSON Schema。"""
    properties: dict[str, Any] = {}
    required: list[str] = []
    type_hints = _type_hints(func)
    for name, parameter in inspect.signature(func).parameters.items():
        if name in preset_kwargs:
            continue
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            raise IrisToolValidationError("工具函数只支持普通参数和关键字参数", parameter=name)
        annotation = type_hints.get(name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            raise IrisToolValidationError("工具函数参数必须包含类型注解", parameter=name)
        field_schema = _schema_for_annotation(annotation)
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            field_schema["default"] = parameter.default
        properties[name] = field_schema
    return {"type": "object", "properties": properties, "required": required}


def callable_input_model(
    func: Callable[..., Any],
    preset_kwargs: set[str],
) -> type[BaseModel]:
    """为 callable 构造输入校验模型。"""
    fields: dict[str, Any] = {}
    type_hints = _type_hints(func)
    for name, parameter in inspect.signature(func).parameters.items():
        if name in preset_kwargs:
            continue
        annotation = type_hints.get(name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            raise IrisToolValidationError("工具函数参数必须包含类型注解", parameter=name)
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[name] = (annotation, Field(default=default))
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


def _schema_for_annotation(annotation: Any) -> dict[str, Any]:
    """将常见 Python 类型注解映射为 JSON Schema。"""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {UnionType, Union}:
        non_none_args = [arg for arg in args if arg is not NoneType]
        if len(non_none_args) == 1 and len(non_none_args) != len(args):
            schema = _schema_for_annotation(non_none_args[0])
            schema["nullable"] = True
            return schema
        return {"anyOf": [_schema_for_annotation(arg) for arg in args]}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if origin is list or annotation is list:
        item_annotation = args[0] if args else Any
        return {"type": "array", "items": _schema_for_annotation(item_annotation)}
    if origin is dict or annotation is dict:
        return {"type": "object"}
    if annotation is Any:
        return {}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return schema_from_pydantic_model(annotation)
    raise IrisToolValidationError("不支持的工具参数类型", annotation=str(annotation))


def _type_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """解析 postponed annotations。"""
    try:
        return get_type_hints(func)
    except NameError as exc:
        raise IrisToolValidationError("工具函数类型注解无法解析", error=str(exc)) from exc
