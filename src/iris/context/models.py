"""上下文系统数据契约。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..exceptions import IrisContextError
from ..message import Msg

_XML_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _is_safe_xml_name(value: str) -> bool:
    return _XML_NAME_RE.fullmatch(value) is not None


class ContextSlot(BaseModel):
    """Context section 内的结构化内容块。"""

    name: str
    content: Any
    order: int = 100
    attributes: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value.strip():
            raise IrisContextError("context slot 名称不能为空")
        if not _is_safe_xml_name(value):
            raise IrisContextError("context slot 名称必须是安全的 XML 标签名")
        return value

    @field_validator("order", mode="before")
    @classmethod
    def _validate_order(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise IrisContextError("context slot 顺序不能是布尔值")
        return value

    @field_validator("attributes")
    @classmethod
    def _validate_attributes(cls, value: dict[str, str]) -> dict[str, str]:
        for key in value:
            if not _is_safe_xml_name(key):
                raise IrisContextError("context slot 属性名必须是安全的 XML 名称")
        return value


class ContextSection(BaseModel):
    """单个固定消息位置的模板、字符上限和 slot。"""

    template: Path | None = None
    max_chars: int | None = None
    slots: list[ContextSlot] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("template")
    @classmethod
    def _validate_template(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise IrisContextError("context 模板路径必须是绝对路径", path=str(value))
        return value

    @field_validator("max_chars", mode="before")
    @classmethod
    def _validate_max_chars_type(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise IrisContextError("context section 字符上限不能是布尔值")
        return value

    @field_validator("max_chars")
    @classmethod
    def _validate_max_chars(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise IrisContextError("context section 字符上限必须为正数")
        return value


class ContextBuildInput(BaseModel):
    """ContextBuilder 的三个固定消息位置输入。"""

    system: ContextSection
    memory: ContextSection | None = None
    before_current_input: ContextSection | None = None

    model_config = ConfigDict(extra="forbid")

    def with_memory_slots(self, *slots: ContextSlot) -> ContextBuildInput:
        """返回追加运行态 memory slots 的新输入。"""
        memory = self.memory or ContextSection()
        updated_memory = memory.model_copy(
            update={"slots": [*memory.slots, *slots]},
        )
        return self.model_copy(update={"memory": updated_memory})

    @model_validator(mode="after")
    def _validate_system_slots(self) -> ContextBuildInput:
        if not any(slot.enabled for slot in self.system.slots):
            raise IrisContextError("system context 必须至少包含一个启用的 slot")
        return self


class ContextBuildOutput(BaseModel):
    """ContextBuilder 生成的三个固定消息位置输出。"""

    system: Msg
    memory: Msg | None = None
    before_current_input: Msg | None = None

    model_config = ConfigDict(extra="forbid")


__all__ = [
    "ContextSlot",
    "ContextSection",
    "ContextBuildInput",
    "ContextBuildOutput",
]
