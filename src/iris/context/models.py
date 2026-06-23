"""上下文系统数据契约。"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..exceptions import IrisContextError
from ..message import Msg

# 字符串必须以字母或下划线开头，后面可以跟字母、数字、下划线、点、短横线，不能有其他任何字符。
_XML_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class ContextPosition(StrEnum):
    """Context 生成消息的固定插入位置。"""

    SYSTEM = "system"
    MEMORY = "memory"
    BEFORE_CURRENT_INPUT = "before_current_input"


class ContextSlot(BaseModel):
    """会被渲染到某个 XML 位置中的结构化上下文片段。"""

    name: str
    position: ContextPosition
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
        if not _XML_NAME_RE.fullmatch(value):
            raise IrisContextError("context slot 名称必须是安全的 XML 标签名")
        return value

    @field_validator("attributes")
    @classmethod
    def _validate_attributes(cls, value: dict[str, str]) -> dict[str, str]:
        for key in value:
            if not _XML_NAME_RE.fullmatch(key):
                raise IrisContextError("context slot 属性名必须是安全的 XML 名称")
        return value


class SystemPromptSpec(BaseModel):
    """System prompt 的来源定义。"""

    mode: Literal["inline", "template"] = "inline"
    inline: str | None = None
    template_path: str | Path | None = None
    variables: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_source(self) -> SystemPromptSpec:
        if self.mode == "inline" and not (self.inline or "").strip():
            raise IrisContextError("inline system prompt 不能为空")
        if self.mode == "template" and self.template_path is None:
            raise IrisContextError("template system prompt 必须提供 template_path")
        return self


class ContextTemplateSpec(BaseModel):
    """每个 context 位置可选的模板路径。"""

    system: str | Path | None = None
    memory: str | Path | None = None
    before_current_input: str | Path | None = None

    model_config = ConfigDict(extra="forbid")


class MemoryContextItem(BaseModel):
    """一条可进入 context 渲染的召回记忆。"""

    text: str
    id: str = ""
    source: str = ""
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise IrisContextError("memory item 文本不能为空")
        return value


class MemoryContextInput(BaseModel):
    """调用方传入的记忆上下文。"""

    enabled: bool = True
    entries: list[MemoryContextItem] = Field(default_factory=list)
    query_from_current_input: bool = False
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ContextBuildInput(BaseModel):
    """ContextBuilder 的输入。

    history、current input message、tools 与 LLMRequest 组装都在本模型之外处理。
    current_input 只作为信号，不会被默认渲染输出。
    """

    agent_id: str
    session_id: str | None = None
    workspace_root: Path | None = None
    template_base_dir: Path | None = None

    system: SystemPromptSpec
    templates: ContextTemplateSpec = Field(default_factory=ContextTemplateSpec)
    memory: MemoryContextInput | None = None
    environment_state: dict[str, Any] = Field(default_factory=dict)
    turn_constraints: list[str] = Field(default_factory=list)
    slots: list[ContextSlot] = Field(default_factory=list)

    current_input: str | None = None
    version: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_id")
    @classmethod
    def _validate_agent_id(cls, value: str) -> str:
        if not value.strip():
            raise IrisContextError("agent_id 不能为空")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: int) -> int:
        if value <= 0:
            raise IrisContextError("context version 必须为正数")
        return value


class ContextBuildOutput(BaseModel):
    """生成出的 API 原生 context 消息增量。"""

    system_message: Msg
    memory_messages: list[Msg] = Field(default_factory=list)
    before_current_input_messages: list[Msg] = Field(default_factory=list)
    slots: list[ContextSlot] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "ContextBuildInput",
    "ContextBuildOutput",
    "ContextPosition",
    "ContextSlot",
    "ContextTemplateSpec",
    "MemoryContextInput",
    "MemoryContextItem",
    "SystemPromptSpec",
]
