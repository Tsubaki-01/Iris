from __future__ import annotations

from pathlib import Path

from iris.context import ContextXmlRenderer
from iris.skill.catalog import (
    CATALOG_SLOT_NAME,
    CATALOG_SLOT_ORDER,
    CATALOG_USAGE_HINT,
    SkillCatalog,
)
from iris.skill.models import SkillDiscoveryResult, SkillMetadata, SkillScope
from iris.skill.registry import SkillRegistry


def _metadata(name: str, description: str, *, root_index: int) -> SkillMetadata:
    root_dir = Path(f"C:/workspace/skills/{name}")
    return SkillMetadata(
        name=name,
        description=description,
        scope=SkillScope.PROJECT,
        skill_file=root_dir / "SKILL.md",
        root_dir=root_dir,
        relative_skill_file=f"skills/{name}/SKILL.md",
        content_version=f"version-{name}",
        root_index=root_index,
        extra_frontmatter={"internal": "hidden"},
    )


def _catalog() -> SkillCatalog:
    registry = SkillRegistry(
        SkillDiscoveryResult(
            skills=(
                _metadata("bravo", "Use <bravo> & helpers", root_index=0),
                _metadata("alpha", "Use alpha", root_index=1),
            ),
            diagnostics=(),
        )
    )
    return SkillCatalog(registry)


def test_catalog_slot_has_exact_structure_and_attributes() -> None:
    slot = _catalog().build_slot()

    assert slot.name == CATALOG_SLOT_NAME == "available_skills"
    assert slot.order == CATALOG_SLOT_ORDER == 900
    assert slot.content == [
        {"name": "bravo", "description": "Use <bravo> & helpers"},
        {"name": "alpha", "description": "Use alpha"},
    ]
    assert slot.attributes == {
        "count": "2",
        "usage": CATALOG_USAGE_HINT,
    }
    assert "chars" not in slot.attributes


def test_catalog_renderer_escapes_special_characters_once_and_sorts_dict_keys() -> None:
    rendered = ContextXmlRenderer().render_slot(_catalog().build_slot())

    assert "Use &lt;bravo&gt; &amp; helpers" in rendered
    assert "&amp;lt;" not in rendered
    assert rendered.index('name="description"') < rendered.index('name="name"')
    assert rendered.index("bravo") < rendered.index("alpha")
    assert "load_skill" in rendered
