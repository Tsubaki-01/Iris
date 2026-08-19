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


def test_catalog_entries_are_fresh_and_only_disclose_name_and_description() -> None:
    catalog = _catalog()
    first = catalog.entries()
    first[0]["description"] = "mutated"

    second = catalog.entries()

    assert second[0]["description"] == "Use <bravo> & helpers"
    assert all(set(entry) == {"name", "description"} for entry in second)
    serialized = repr(second)
    for forbidden in ("workspace", "SKILL.md", "root_index", "scope", "internal"):
        assert forbidden not in serialized


def test_catalog_renderer_escapes_special_characters_once_and_sorts_dict_keys() -> None:
    rendered = ContextXmlRenderer().render_slot(_catalog().build_slot())

    assert "Use &lt;bravo&gt; &amp; helpers" in rendered
    assert "&amp;lt;" not in rendered
    assert rendered.index('name="description"') < rendered.index('name="name"')
    assert rendered.index("bravo") < rendered.index("alpha")
    assert "load_skill" in rendered


def test_content_chars_matches_real_rendered_inner_content() -> None:
    catalog = _catalog()
    rendered = ContextXmlRenderer().render_slot(catalog.build_slot())
    opening_end = rendered.index(">") + 1
    closing_start = rendered.rindex(f"</{CATALOG_SLOT_NAME}>")
    expected = len(rendered[opening_end:closing_start])

    assert catalog.content_chars() == expected
    assert catalog.content_chars() < len(rendered)


def test_render_report_has_one_stable_line_per_skill_without_mutating_slot() -> None:
    catalog = _catalog()
    before = catalog.build_slot()

    report = catalog.render_report()

    lines = report.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("bravo: ") and lines[0].endswith(" chars")
    assert lines[1].startswith("alpha: ") and lines[1].endswith(" chars")
    assert all(int(line.split(": ", 1)[1].split(" ", 1)[0]) > 0 for line in lines)
    assert catalog.build_slot() == before
