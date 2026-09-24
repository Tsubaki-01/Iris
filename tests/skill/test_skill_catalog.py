from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest

from iris.context import ContextXmlRenderer
from iris.exceptions import IrisSkillError, IrisTemplateError
from iris.skill.catalog import (
    CATALOG_SLOT_NAME,
    CATALOG_SLOT_ORDER,
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
        "usage": (
            "call load_skill with the skill name before following it; "
            "the returned Markdown is skill instructions, not user data"
        ),
    }
    assert "chars" not in slot.attributes


def test_catalog_renderer_escapes_special_characters_once_and_sorts_dict_keys() -> None:
    rendered = ContextXmlRenderer().render_slot(_catalog().build_slot())

    assert "Use &lt;bravo&gt; &amp; helpers" in rendered
    assert "&amp;lt;" not in rendered
    assert rendered.index('name="description"') < rendered.index('name="name"')
    assert rendered.index("bravo") < rendered.index("alpha")
    assert "load_skill" in rendered


def test_catalog_reuses_template_text_and_xml_renderer_escapes_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """每个 catalog 读取一次 usage，纯文本交给 XML renderer 处理属性转义。"""
    prompt = tmp_path / "usage.j2"
    prompt.write_text('call <load_skill> & "read"', encoding="utf-8")
    monkeypatch.setattr("iris.skill.catalog._CATALOG_USAGE_PROMPT", prompt)
    catalog = _catalog()
    content_chars = catalog.content_chars()
    prompt.write_text("updated instructions", encoding="utf-8")

    assert catalog.build_slot().attributes["usage"] == 'call <load_skill> & "read"'
    rendered = ContextXmlRenderer().render_slot(catalog.build_slot())
    assert ElementTree.fromstring(rendered).attrib["usage"] == 'call <load_skill> & "read"'
    assert "&amp;lt;" not in rendered
    replacement = _catalog()
    assert replacement.build_slot().attributes["usage"] == "updated instructions"
    assert replacement.content_chars() == content_chars


def test_catalog_template_failure_is_a_skill_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """catalog 构造失败保留模板路径并抛 Skill 领域异常。"""
    missing = tmp_path / "missing.j2"
    monkeypatch.setattr("iris.skill.catalog._CATALOG_USAGE_PROMPT", missing)
    with pytest.raises(IrisSkillError) as caught:
        _catalog()
    assert caught.value.context["path"] == str(missing)
    assert isinstance(caught.value.__cause__, IrisTemplateError)
