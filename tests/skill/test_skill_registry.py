from __future__ import annotations

from pathlib import Path

import pytest

import iris.skill as skill_package
from iris.exceptions import IrisSkillError, IrisSkillNotFoundError
from iris.skill.models import (
    SkillDiagnostic,
    SkillDiscoveryResult,
    SkillMetadata,
    SkillScope,
)
from iris.skill.registry import SkillRegistry


def _metadata(name: str, *, root_index: int = 0) -> SkillMetadata:
    root_dir = Path(f"C:/workspace/skills/{name}")
    return SkillMetadata(
        name=name,
        description=f"Use {name}",
        scope=SkillScope.PROJECT,
        skill_file=root_dir / "SKILL.md",
        root_dir=root_dir,
        relative_skill_file=f"skills/{name}/SKILL.md",
        root_index=root_index,
    )


def test_registry_preserves_discovery_order_and_diagnostics() -> None:
    diagnostic = SkillDiagnostic(code="warning", message="Example warning")
    bravo = _metadata("bravo", root_index=0)
    alpha = _metadata("alpha", root_index=1)
    registry = SkillRegistry(
        SkillDiscoveryResult(
            skills=(bravo, alpha),
            diagnostics=(diagnostic,),
        )
    )

    assert len(registry) == 2
    assert registry.names() == ("bravo", "alpha")
    assert registry.get("bravo") is bravo
    assert registry.get("alpha") is alpha
    assert registry.has("bravo") is True
    assert registry.has("missing") is False
    assert registry.diagnostics == (diagnostic,)


def test_registry_missing_get_raises_domain_error_with_available_names() -> None:
    registry = SkillRegistry(
        SkillDiscoveryResult(skills=(_metadata("alpha"),), diagnostics=())
    )

    with pytest.raises(IrisSkillNotFoundError, match="missing") as exc_info:
        registry.get("missing")

    assert exc_info.value.context == {
        "name": "missing",
        "available": ("alpha",),
    }


def test_registry_reports_missing_names_stably_without_duplicates() -> None:
    registry = SkillRegistry(
        SkillDiscoveryResult(
            skills=(_metadata("alpha"), _metadata("bravo")),
            diagnostics=(),
        )
    )

    assert registry.missing(["alpha", "nope", "nope", "other", "bravo"]) == (
        "nope",
        "other",
    )


def test_registry_rejects_duplicate_names_in_discovery_result() -> None:
    with pytest.raises(IrisSkillError, match="重复"):
        SkillRegistry(
            SkillDiscoveryResult(
                skills=(_metadata("alpha"), _metadata("alpha", root_index=1)),
                diagnostics=(),
            )
        )


def test_skill_package_exports_only_planned_public_surface() -> None:
    assert set(skill_package.__all__) == {
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
    }
    assert not hasattr(skill_package, "LoadSkillInput")
    assert not hasattr(skill_package, "parse_frontmatter")
