"""模板遵循 Jinja 原生按需加载、编译缓存与文件更新语义。"""

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from iris.context import ContextTemplateRenderer
from iris.exceptions import IrisContextError


def _rewrite(path: Path, source: str) -> None:
    """推进 mtime，让修改检测不依赖文件系统时钟精度。"""
    modified = path.stat().st_mtime + 2
    path.write_text(source, encoding="utf-8")
    os.utime(path, (modified, modified))


def test_renderer_reloads_entry_and_nested_dependencies(tmp_path: Path) -> None:
    """同一 renderer 读取入口和 extends/import/include 依赖的更新。"""
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
    _rewrite(macros, "{% macro content(x) %}new {{ x }}{% endmacro %}")
    assert renderer.render_file(main, {"value": "second"}) == "<root>new second</root>"
    _rewrite(base, "<changed>{% block body %}{% endblock %}</changed>")
    _rewrite(body, 'body {% import "macros.j2" as m %}{{ m.content(value) }}')
    assert renderer.render_file(main, {"value": "third"}) == "<changed>body new third</changed>"
    _rewrite(main, "entry {{ value }}")
    assert renderer.render_file(main, {"value": "fourth"}) == "entry fourth"


def test_renderer_loads_added_optional_include(tmp_path: Path) -> None:
    main = tmp_path / "main.j2"
    main.write_text('{% include "optional.j2" ignore missing %}base', encoding="utf-8")
    renderer = ContextTemplateRenderer()
    assert renderer.render_file(main, {}) == "base"
    (tmp_path / "optional.j2").write_text("added ", encoding="utf-8")
    assert renderer.render_file(main, {}) == "added base"


def test_renderer_loads_dependency_only_when_branch_is_rendered(tmp_path: Path) -> None:
    main = tmp_path / "main.j2"
    included = tmp_path / "later.j2"
    main.write_text('{% if use_later %}{% include "later.j2" %}{% endif %}base', encoding="utf-8")
    included.write_text("{% broken syntax %}", encoding="utf-8")
    renderer = ContextTemplateRenderer()
    assert renderer.render_file(main, {"use_later": False}) == "base"
    with pytest.raises(IrisContextError):
        renderer.render_file(main, {"use_later": True})
    _rewrite(included, "changed ")
    assert renderer.render_file(main, {"use_later": True}) == "changed base"


def test_renderer_loads_new_preferred_include_candidate(tmp_path: Path) -> None:
    main = tmp_path / "main.j2"
    main.write_text('{% include ["preferred.j2", "fallback.j2"] %}', encoding="utf-8")
    (tmp_path / "fallback.j2").write_text("fallback", encoding="utf-8")
    renderer = ContextTemplateRenderer()
    assert renderer.render_file(main, {}) == "fallback"
    (tmp_path / "preferred.j2").write_text("preferred", encoding="utf-8")
    assert renderer.render_file(main, {}) == "preferred"


@pytest.mark.parametrize(
    "source, expected",
    [
        ("{% include selected %}", "selected"),
        ("{% extends selected %}", "selected"),
        ("{% import selected as m %}{{ m.content() }}", "macro"),
    ],
)
def test_renderer_accepts_dynamic_template_filenames(
    tmp_path: Path, source: str, expected: str
) -> None:
    main = tmp_path / "main.j2"
    main.write_text(source, encoding="utf-8")
    (tmp_path / "selected.j2").write_text(
        "selected{% macro content() %}macro{% endmacro %}", encoding="utf-8"
    )
    assert ContextTemplateRenderer().render_file(main, {"selected": "selected.j2"}) == expected


def test_renderer_reuses_compiled_template_with_current_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main = tmp_path / "main.j2"
    main.write_text("{{ value }}", encoding="utf-8")
    loaded: list[str] = []
    original = FileSystemLoader.get_source

    def get_source(
        self: FileSystemLoader, environment: Environment, template: str
    ) -> tuple[str, str | None, Callable[[], bool] | None]:
        loaded.append(template)
        return original(self, environment, template)

    monkeypatch.setattr(FileSystemLoader, "get_source", get_source)
    renderer = ContextTemplateRenderer()
    assert renderer.render_file(main, {"value": "first"}) == "first"
    assert renderer.render_file(main, {"value": "<&>"}) == "&lt;&amp;&gt;"
    with pytest.raises(IrisContextError):
        renderer.render_file(main, {})
    assert loaded == ["main.j2"]


def test_renderer_isolates_same_names_in_different_directories(tmp_path: Path) -> None:
    renderer = ContextTemplateRenderer()
    for name in ("first", "second", "first"):
        directory = tmp_path / name
        directory.mkdir(exist_ok=True)
        main = directory / "main.j2"
        main.write_text('{% include "body.j2" %}', encoding="utf-8")
        (directory / "body.j2").write_text(name, encoding="utf-8")
        assert renderer.render_file(main, {}) == name


@pytest.mark.parametrize("target", ["entry", "dependency"])
def test_renderer_reports_changed_invalid_or_deleted_files(tmp_path: Path, target: str) -> None:
    main = tmp_path / "main.j2"
    dependency = tmp_path / "body.j2"
    main.write_text('{% include "body.j2" %}', encoding="utf-8")
    dependency.write_text("original", encoding="utf-8")
    changed = main if target == "entry" else dependency
    renderer = ContextTemplateRenderer()
    assert renderer.render_file(main, {}) == "original"
    _rewrite(changed, "{% broken syntax %}")
    with pytest.raises(IrisContextError):
        renderer.render_file(main, {})
    _rewrite(changed, "corrected")
    assert renderer.render_file(main, {}) == "corrected"
    changed.unlink()
    with pytest.raises(IrisContextError):
        renderer.render_file(main, {})
