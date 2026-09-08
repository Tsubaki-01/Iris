"""模板渲染复用首次读取的来源与编译结果。"""

from pathlib import Path

import pytest

from iris.context import ContextTemplateRenderer
from iris.exceptions import IrisContextError


def test_renderer_keeps_nested_template_sources_until_recreated(tmp_path: Path) -> None:
    """模板 extends、import 和 include 的内容在同一 renderer 内保持一致。"""
    main = tmp_path / "main.j2"
    base = tmp_path / "base.j2"
    macros = tmp_path / "macros.j2"
    body = tmp_path / "body.j2"
    main.write_text(
        '{% extends "base.j2" %}{% block body %}{% include "body.j2" %}{% endblock %}',
        encoding="utf-8",
    )
    base.write_text("<root>{% block body %}{% endblock %}</root>", encoding="utf-8")
    body.write_text('{% import "macros.j2" as m %}{{ m.content(value) }}', encoding="utf-8")
    macros.write_text("{% macro content(x) %}old {{ x }}{% endmacro %}", encoding="utf-8")
    renderer = ContextTemplateRenderer()
    assert renderer.render_file(main, {"value": "<&>"}) == "<root>old &lt;&amp;&gt;</root>"
    macros.write_text("{% macro content(x) %}new {{ x }}{% endmacro %}", encoding="utf-8")
    assert renderer.render_file(main, {"value": "second"}) == "<root>old second</root>"
    assert ContextTemplateRenderer().render_file(main, {"value": "second"}) == (
        "<root>new second</root>"
    )


def test_renderer_freezes_missing_optional_include(tmp_path: Path) -> None:
    """可选 include 在快照中缺失时，随后新文件只影响新 renderer。"""
    main = tmp_path / "main.j2"
    main.write_text('{% include "optional.j2" ignore missing %}base', encoding="utf-8")
    renderer = ContextTemplateRenderer()
    assert renderer.render_file(main, {}) == "base"
    (tmp_path / "optional.j2").write_text("added ", encoding="utf-8")
    assert renderer.render_file(main, {}) == "base"
    assert ContextTemplateRenderer().render_file(main, {}) == "added base"


def test_renderer_freezes_static_include_before_branch_is_rendered(tmp_path: Path) -> None:
    """尚未进入的静态分支也绑定首次来源，后续选择分支不会换规则。"""
    main = tmp_path / "main.j2"
    included = tmp_path / "later.j2"
    main.write_text('{% if use_later %}{% include "later.j2" %}{% endif %}base', encoding="utf-8")
    included.write_text("original ", encoding="utf-8")
    renderer = ContextTemplateRenderer()
    assert renderer.render_file(main, {"use_later": False}) == "base"
    included.write_text("changed ", encoding="utf-8")
    assert renderer.render_file(main, {"use_later": True}) == "original base"


def test_renderer_freezes_include_fallback_candidates(tmp_path: Path) -> None:
    """列表候选保留首次存在性与内容，新文件不能改变同一快照的选择。"""
    main = tmp_path / "main.j2"
    main.write_text('{% include ["preferred.j2", "fallback.j2"] %}', encoding="utf-8")
    (tmp_path / "fallback.j2").write_text("fallback", encoding="utf-8")
    renderer = ContextTemplateRenderer()
    assert renderer.render_file(main, {}) == "fallback"
    (tmp_path / "preferred.j2").write_text("preferred", encoding="utf-8")
    assert renderer.render_file(main, {}) == "fallback"
    assert ContextTemplateRenderer().render_file(main, {}) == "preferred"


@pytest.mark.parametrize(
    "source",
    ["{% include selected %}", "{% extends selected %}", "{% import selected as m %}"],
)
def test_renderer_rejects_dynamic_template_filenames(tmp_path: Path, source: str) -> None:
    """动态数据仍可渲染，但无法启动冻结的动态文件名提供明确修正指引。"""
    main = tmp_path / "main.j2"
    main.write_text(source, encoding="utf-8")
    (tmp_path / "selected.j2").write_text("selected", encoding="utf-8")
    with pytest.raises(IrisContextError, match="静态文件名"):
        ContextTemplateRenderer().render_file(main, {"selected": "selected.j2"})
