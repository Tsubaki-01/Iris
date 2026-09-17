"""Skill 子系统的领域模型。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

SKILL_FILE_NAME = "SKILL.md"
DEFAULT_SKILLS_ROOT = ".agents/skills"
MAX_DESCRIPTION_CHARS = 1024
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillScope(StrEnum):
    """Skill 的发现范围。"""

    PROJECT = "project"


def _normalize_description(value: str, max_chars: int) -> tuple[str, bool]:
    """规范化 description，并返回是否发生截断。"""
    normalized = value.strip()
    if not normalized:
        raise ValueError("description 不能为空")
    truncated = len(normalized) > max_chars
    return normalized[:max_chars], truncated


class SkillMetadata(BaseModel):
    """不包含正文的单条 Skill 元数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    scope: SkillScope
    skill_file: Path
    root_dir: Path
    relative_skill_file: str
    root_index: int = Field(ge=0)
    description_truncated: bool = False
    declared_name: str | None = None
    extra_frontmatter: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SkillDiagnostic:
    """Skill 发现期产生的结构化 warning。"""

    code: str
    message: str
    path: Path | None = None
    detail: dict[str, str] = field(default_factory=dict)


class SkillDiscoveryOptions(BaseModel):
    """Skill discovery 的固定输入选项。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_root: Path
    roots: tuple[tuple[SkillScope, Path], ...]
    max_description_chars: int = Field(
        default=MAX_DESCRIPTION_CHARS,
        strict=True,
        gt=0,
    )


class SkillDiscoveryResult(BaseModel):
    """Skill discovery 的稳定结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    skills: tuple[SkillMetadata, ...]
    diagnostics: tuple[SkillDiagnostic, ...]


__all__ = [
    "DEFAULT_SKILLS_ROOT",
    "MAX_DESCRIPTION_CHARS",
    "SKILL_FILE_NAME",
    "SkillDiagnostic",
    "SkillDiscoveryOptions",
    "SkillDiscoveryResult",
    "SkillMetadata",
    "SkillScope",
]
