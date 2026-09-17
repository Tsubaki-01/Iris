"""Lifecycle store 共用的私有序列化投影。

该模块只把已经验证的 command 值转换为 durable JSON 可消费的稳定值。

Example:
    payload = jsonable(command)
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum

from pydantic import BaseModel


def jsonable(value: object) -> object:
    """把 lifecycle command 投影为稳定 JSON 值。"""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
