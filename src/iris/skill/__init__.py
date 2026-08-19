"""Iris Skill 子系统公共 API。"""

from .catalog import CATALOG_SLOT_NAME, SkillCatalog
from .discovery import discover_skills, resolve_skills_root
from .models import (
    DEFAULT_SKILLS_ROOT,
    SkillDiagnostic,
    SkillDiscoveryOptions,
    SkillDiscoveryResult,
    SkillMetadata,
    SkillScope,
)
from .registry import SkillRegistry
from .tool import LoadSkillTool

__all__ = [
    "CATALOG_SLOT_NAME",
    "DEFAULT_SKILLS_ROOT",
    "LoadSkillTool",
    "SkillCatalog",
    "SkillDiagnostic",
    "SkillDiscoveryOptions",
    "SkillDiscoveryResult",
    "SkillMetadata",
    "SkillRegistry",
    "SkillScope",
    "discover_skills",
    "resolve_skills_root",
]
