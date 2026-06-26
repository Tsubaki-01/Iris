from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import iris.context as context
from iris.context.config import _is_unsupported_windows_template_path
from iris.exceptions import IrisContextError


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("templates/x.j2", False),
        (r"\templates\x.j2", True),
        (r"C:templates\x.j2", True),
        (r"C:\templates\x.j2", True),
    ],
)
def test_windows_template_path_classification_is_cross_platform(
    template: str,
    expected: bool,
) -> None:
    assert (
        _is_unsupported_windows_template_path(
            template,
            host_is_absolute=False,
        )
        is expected
    )


def test_load_context_build_input_loads_sections_and_resolves_templates(
    tmp_path: Path,
) -> None:
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    system_template = templates_dir / "system.xml.j2"
    memory_template = templates_dir / "memory.xml.j2"
    before_template = templates_dir / "before.xml.j2"
    system_template.write_text(
        "<system_context>"
        "{% for slot in slots %}"
        "<{{ slot.name }}>{{ slot.content }}</{{ slot.name }}>"
        "{% endfor %}"
        "</system_context>",
        encoding="utf-8",
    )
    memory_template.write_text("<memory_context />", encoding="utf-8")
    before_template.write_text(
        "<before_current_input_context />",
        encoding="utf-8",
    )
    config_path = tmp_path / "context.yaml"
    config_path.write_text(
        f"""
system:
  template: templates/system.xml.j2
  max_chars: 500
  slots:
    - name: instructions
      content: system content
      order: 20
      attributes:
        source: local
      enabled: true
    - name: disabled
      content: hidden
      order: 1
      enabled: false
memory:
  template: {memory_template.resolve().as_posix()}
  max_chars: 300
  slots:
    - name: memory
      content: memory content
before_current_input:
  template: templates/before.xml.j2
  max_chars: 200
  slots:
    - name: environment_state
      content: input content
""".strip(),
        encoding="utf-8",
    )

    result = context.load_context_build_input(config_path)

    assert result.system.template == system_template.resolve()
    assert result.memory is not None
    assert result.memory.template == memory_template.resolve()
    assert result.before_current_input is not None
    assert result.before_current_input.template == before_template.resolve()
    assert result.system.template.is_absolute()
    assert result.memory.template.is_absolute()
    assert result.before_current_input.template.is_absolute()
    assert result.system.max_chars == 500
    assert result.memory.max_chars == 300
    assert result.before_current_input.max_chars == 200
    assert result.system.slots[0].content == "system content"
    assert result.system.slots[0].order == 20
    assert result.system.slots[0].attributes == {"source": "local"}
    assert result.system.slots[0].enabled is True
    assert result.system.slots[1].enabled is False
    assert result.memory.slots[0].content == "memory content"
    assert result.before_current_input.slots[0].content == "input content"

    output = context.ContextBuilder().build(result)

    assert output.system.text == (
        "<system_context><instructions>system content</instructions></system_context>"
    )


def test_runtime_slot_can_be_added_with_helper_without_mutating_input(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "context.yaml"
    config_path.write_text(
        """
system:
  slots:
    - name: instructions
      content: system content
memory:
  slots:
    - name: configured_memory
      content: configured content
""".strip(),
        encoding="utf-8",
    )
    loaded = context.load_context_build_input(config_path)
    assert loaded.memory is not None
    runtime_slot = context.ContextSlot(
        name="runtime_memory",
        content="runtime content",
    )

    runtime_input = loaded.with_memory_slots(runtime_slot)
    output = context.ContextBuilder().build(runtime_input)

    assert output.memory is not None
    assert "runtime content" in output.memory.text
    assert len(loaded.memory.slots) == 1
    assert loaded.memory.slots[0].name == "configured_memory"


def test_runtime_memory_slot_helper_creates_missing_memory_section(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "context.yaml"
    config_path.write_text(
        """
system:
  slots:
    - name: instructions
      content: system content
""".strip(),
        encoding="utf-8",
    )
    loaded = context.load_context_build_input(config_path)
    runtime_slot = context.ContextSlot(
        name="runtime_memory",
        content="runtime content",
        attributes={"source": "runtime"},
    )

    runtime_input = loaded.with_memory_slots(runtime_slot)
    output = context.ContextBuilder().build(runtime_input)

    assert loaded.memory is None
    assert output.memory is not None
    assert (
        '<runtime_memory source="runtime">runtime content</runtime_memory>'
        in output.memory.text
    )


@pytest.mark.parametrize(
    "legacy_yaml",
    [
        "version: 1",
        "templates: {}",
        "metadata: {}",
        "before_current_input:\n  environment_state: legacy",
        (
            "system:\n"
            "  slots:\n"
            "    - name: instructions\n"
            "      content: content\n"
            "      position: system"
        ),
    ],
)
def test_load_context_build_input_rejects_legacy_fields(
    tmp_path: Path,
    legacy_yaml: str,
) -> None:
    config_path = tmp_path / "context.yaml"
    if legacy_yaml.startswith("system:"):
        yaml_text = legacy_yaml
    else:
        yaml_text = (
            "system:\n"
            "  slots:\n"
            "    - name: instructions\n"
            "      content: content\n"
            f"{legacy_yaml}\n"
        )
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(IrisContextError) as exc_info:
        context.load_context_build_input(config_path)

    assert exc_info.value.context["path"] == str(config_path)
    assert exc_info.value.context["error"]


def test_load_context_build_input_rejects_inline_with_valid_system_slot(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "context.yaml"
    config_path.write_text(
        (
            "system:\n"
            "  inline: legacy\n"
            "  slots:\n"
            "    - name: instructions\n"
            "      content: content\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(IrisContextError) as exc_info:
        context.load_context_build_input(config_path)

    assert exc_info.value.context["path"] == str(config_path)
    assert "inline" in exc_info.value.context["error"]


@pytest.mark.parametrize(
    "template",
    [
        r"\templates\system.xml.j2",
        r"C:templates\system.xml.j2",
    ],
)
def test_load_context_build_input_rejects_anchored_non_absolute_template(
    tmp_path: Path,
    template: str,
) -> None:
    config_path = tmp_path / "context.yaml"
    config_path.write_text(
        (
            "system:\n"
            f"  template: '{template}'\n"
            "  slots:\n"
            "    - name: instructions\n"
            "      content: content\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(IrisContextError) as exc_info:
        context.load_context_build_input(config_path)

    assert exc_info.value.context["path"] == str(config_path)
    assert exc_info.value.context["section"] == "system"
    assert exc_info.value.context["template"] == template


def test_load_context_build_input_wraps_template_path_resolution_error(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "context.yaml"
    template = "bad\x00path"
    config_path.write_text(
        (
            "system:\n"
            '  template: "bad\\0path"\n'
            "  slots:\n"
            "    - name: instructions\n"
            "      content: content\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(IrisContextError) as exc_info:
        context.load_context_build_input(config_path)

    assert exc_info.value.context["path"] == str(config_path)
    assert exc_info.value.context["section"] == "system"
    assert exc_info.value.context["template"] == template
    assert exc_info.value.context["error"]


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("non-object.yaml", "- item"),
        ("invalid.yaml", "system: ["),
    ],
)
def test_load_context_build_input_wraps_yaml_errors(
    tmp_path: Path,
    filename: str,
    content: str,
) -> None:
    config_path = tmp_path / filename
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(IrisContextError) as exc_info:
        context.load_context_build_input(config_path)

    assert exc_info.value.context["path"] == str(config_path)


def test_load_context_build_input_wraps_missing_file(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.yaml"

    with pytest.raises(IrisContextError) as exc_info:
        context.load_context_build_input(config_path)

    assert exc_info.value.context["path"] == str(config_path)


def test_load_context_build_input_wraps_encoding_error(tmp_path: Path) -> None:
    config_path = tmp_path / "context.yaml"
    config_path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(IrisContextError) as exc_info:
        context.load_context_build_input(config_path)

    assert exc_info.value.context["path"] == str(config_path)


@pytest.mark.parametrize("template", [["system.xml.j2"], 123])
def test_load_context_build_input_wraps_invalid_template_type(
    tmp_path: Path,
    template: object,
) -> None:
    config_path = tmp_path / "context.yaml"
    config_path.write_text(
        (
            "system:\n"
            f"  template: {template!r}\n"
            "  slots:\n"
            "    - name: instructions\n"
            "      content: content\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(IrisContextError) as exc_info:
        context.load_context_build_input(config_path)

    assert exc_info.value.context["path"] == str(config_path)
    assert exc_info.value.context["error"]


def test_load_context_build_input_only_accepts_path_parameter() -> None:
    parameters = inspect.signature(context.load_context_build_input).parameters

    assert list(parameters) == ["path"]
