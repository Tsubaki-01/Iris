from __future__ import annotations

from pathlib import Path

from iris.context import ContextBuilder, load_context_build_input


def test_load_context_build_input_loads_sections_and_resolves_templates(
    tmp_path: Path,
) -> None:
    template_path = tmp_path / "templates" / "context.j2"
    template_path.parent.mkdir()
    template_path.write_text("<context>{{ slots[0].content }}</context>", encoding="utf-8")
    absolute_template_path = tmp_path / "memory.j2"
    absolute_template_path.write_text("<memory>{{ slots[0].content }}</memory>", encoding="utf-8")
    config_path = tmp_path / "context.yaml"
    config_path.write_text(
        "\n".join(
            [
                "system:",
                "  template: templates/context.j2",
                "  slots:",
                "    - name: instructions",
                "      content: system content",
                "memory:",
                f"  template: {absolute_template_path.as_posix()}",
                "  slots:",
                "    - name: memory",
                "      content: memory content",
                "before_current_input:",
                "  template: templates/context.j2",
                "  slots:",
                "    - name: state",
                "      content: input content",
            ]
        ),
        encoding="utf-8",
    )

    result = load_context_build_input(config_path)
    output = ContextBuilder().build(result)

    assert result.system.template == template_path.resolve()
    assert result.memory is not None
    assert result.memory.template == absolute_template_path
    assert result.before_current_input is not None
    assert result.before_current_input.template == template_path.resolve()
    assert output.system.text == "<context>system content</context>"
