"""context.yaml 配置加载与转换。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..exceptions import IrisContextError
from .models import (
    ContextBuildInput,
    ContextSlot,
    ContextTemplateSpec,
    MemoryContextInput,
    MemoryContextItem,
    SystemPromptSpec,
)


class SystemContentConfig(BaseModel):
    """context.yaml 中的 system 内容配置。"""

    mode: Literal["inline", "template"] | None = None
    inline: str | None = None
    template_path: str | Path | None = None
    identity: str = ""
    behavior_rules: list[str] = Field(default_factory=list)
    response_style: list[str] = Field(default_factory=list)
    capability_boundary: list[str] = Field(default_factory=list)
    variables: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    def to_variables(self) -> dict[str, Any]:
        """转换为模板可使用的 system 变量。"""
        variables = {
            "identity": self.identity,
            "behavior_rules": list(self.behavior_rules),
            "response_style": list(self.response_style),
            "capability_boundary": list(self.capability_boundary),
            "inline": self.inline or "",
        }
        variables.update(self.variables)
        return variables

    def to_inline_text(self) -> str:
        """转换为 inline system 文本。"""
        if self.inline and self.inline.strip():
            return self.inline
        sections: list[str] = []
        if self.identity.strip():
            sections.append(self.identity.strip())
        sections.extend(_format_lines("行为规则", self.behavior_rules))
        sections.extend(_format_lines("输出风格", self.response_style))
        sections.extend(_format_lines("能力边界", self.capability_boundary))
        return "\n".join(sections).strip()


class BeforeCurrentInputConfig(BaseModel):
    """current input 前置上下文配置。"""

    environment_state: dict[str, Any] = Field(default_factory=dict)
    turn_constraints: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ContextYamlConfig(BaseModel):
    """context.yaml 顶层配置。"""

    version: int = 1
    templates: ContextTemplateSpec = Field(default_factory=ContextTemplateSpec)
    system: SystemContentConfig
    memory: MemoryContextInput | None = None
    before_current_input: BeforeCurrentInputConfig = Field(
        default_factory=BeforeCurrentInputConfig
    )
    slots: list[ContextSlot] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: int) -> int:
        if value <= 0:
            raise IrisContextError("context.yaml version 必须为正数")
        return value

    def to_build_input(
        self,
        *,
        agent_id: str,
        session_id: str | None = None,
        workspace_root: Path | None = None,
        template_base_dir: Path | None = None,
        memory_items: list[MemoryContextItem] | None = None,
        environment_state: dict[str, Any] | None = None,
        turn_constraints: list[str] | None = None,
        current_input: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ContextBuildInput:
        """转换为 `ContextBuilder` 可直接使用的输入。"""
        return ContextBuildInput(
            agent_id=agent_id,
            session_id=session_id,
            workspace_root=workspace_root,
            template_base_dir=template_base_dir,
            system=self._system_prompt_spec(),
            templates=self._template_spec_for_build_input(),
            memory=self._memory_input(memory_items),
            environment_state=self._merged_environment_state(environment_state),
            turn_constraints=self._merged_turn_constraints(turn_constraints),
            slots=list(self.slots),
            current_input=current_input,
            version=self.version,
            metadata={**self.metadata, **(metadata or {})},
        )

    def _system_prompt_spec(self) -> SystemPromptSpec:
        mode = self.system.mode
        template_path = self.system.template_path or self.templates.system
        if mode == "template" or (mode is None and template_path is not None):
            return SystemPromptSpec(
                mode="template",
                template_path=template_path,
                variables=self.system.to_variables(),
            )
        return SystemPromptSpec(
            mode="inline",
            inline=self.system.to_inline_text(),
            variables=self.system.to_variables(),
        )

    def _template_spec_for_build_input(self) -> ContextTemplateSpec:
        return ContextTemplateSpec(
            system=None,
            memory=self.templates.memory,
            before_current_input=self.templates.before_current_input,
        )

    def _memory_input(
        self,
        memory_items: list[MemoryContextItem] | None,
    ) -> MemoryContextInput | None:
        if self.memory is None and memory_items is None:
            return None
        base = self.memory or MemoryContextInput()
        return base.model_copy(
            update={"entries": [*base.entries, *(memory_items or [])]}
        )

    def _merged_environment_state(
        self,
        environment_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            **self.before_current_input.environment_state,
            **(environment_state or {}),
        }

    def _merged_turn_constraints(self, turn_constraints: list[str] | None) -> list[str]:
        return [
            *self.before_current_input.turn_constraints,
            *(turn_constraints or []),
        ]


def load_context_config(path: str | Path) -> ContextYamlConfig:
    """读取并校验 context.yaml。"""
    config_path = Path(path)
    if not config_path.exists():
        raise IrisContextError("context 配置文件不存在", path=str(config_path))
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IrisContextError(
            "context 配置文件读取失败", path=str(config_path)
        ) from exc
    try:
        raw_config = yaml.safe_load(config_text)
    except yaml.YAMLError as exc:
        raise IrisContextError(
            "context 配置 YAML 解析失败", path=str(config_path)
        ) from exc
    if not isinstance(raw_config, dict):
        raise IrisContextError("context 配置顶层必须是对象", path=str(config_path))
    try:
        return ContextYamlConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise IrisContextError(
            "context 配置校验失败",
            path=str(config_path),
            error=str(exc),
        ) from exc


def load_context_build_input(
    path: str | Path,
    *,
    agent_id: str,
    session_id: str | None = None,
    workspace_root: Path | None = None,
    template_base_dir: Path | None = None,
    memory_items: list[MemoryContextItem] | None = None,
    environment_state: dict[str, Any] | None = None,
    turn_constraints: list[str] | None = None,
    current_input: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ContextBuildInput:
    """读取 context.yaml 并转换为 `ContextBuildInput`。"""
    config_path = Path(path)
    config = load_context_config(config_path)
    return config.to_build_input(
        agent_id=agent_id,
        session_id=session_id,
        workspace_root=workspace_root,
        template_base_dir=template_base_dir or config_path.parent,
        memory_items=memory_items,
        environment_state=environment_state,
        turn_constraints=turn_constraints,
        current_input=current_input,
        metadata=metadata,
    )


def _format_lines(title: str, lines: list[str]) -> list[str]:
    if not lines:
        return []
    return [f"{title}:", *(f"- {line}" for line in lines)]


__all__ = [
    "BeforeCurrentInputConfig",
    "ContextYamlConfig",
    "SystemContentConfig",
    "load_context_build_input",
    "load_context_config",
]
