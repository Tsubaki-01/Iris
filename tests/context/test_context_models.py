from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

import iris.context as context
import iris.context.config as context_config
import iris.context.models as context_models
from iris.context import ContextBuildInput, ContextSection, ContextSlot


def test_context_package_only_exports_section_api() -> None:
    expected_exports = {
        "CONTEXT_SENDER",
        "ContextBuildInput",
        "ContextBuildOutput",
        "ContextBuilder",
        "ContextSection",
        "ContextSlot",
        "ContextTemplateRenderer",
        "ContextXmlRenderer",
        "load_context_build_input",
    }
    assert set(context.__all__) == expected_exports
    for name in expected_exports:
        assert getattr(context, name) is not None


@pytest.mark.parametrize(
    "legacy_attribute",
    [
        "BeforeCurrentInputConfig",
        "ContextBudgetPolicy",
        "ContextPosition",
        "ContextTemplateSpec",
        "MemoryContextInput",
        "MemoryContextItem",
        "SystemContentConfig",
        "SystemPromptSpec",
        "ContextYamlConfig",
        "load_context_config",
    ],
)
def test_context_package_does_not_expose_legacy_api(
    legacy_attribute: str,
) -> None:
    assert not hasattr(context, legacy_attribute)


def test_context_models_only_export_section_contract() -> None:
    assert set(context_models.__all__) == {
        "ContextSlot",
        "ContextSection",
        "ContextBuildInput",
        "ContextBuildOutput",
    }


def test_legacy_budget_module_is_removed() -> None:
    assert importlib.util.find_spec("iris.context.budget") is None


@pytest.mark.parametrize(
    "legacy_attribute",
    [
        "ContextYamlConfig",
        "BeforeCurrentInputConfig",
        "SystemContentConfig",
        "load_context_config",
    ],
)
def test_context_config_does_not_expose_legacy_api(
    legacy_attribute: str,
) -> None:
    assert not hasattr(context_config, legacy_attribute)


@pytest.mark.parametrize(
    "legacy_field",
    [
        {"position": "system"},
        {"unknown": "value"},
    ],
)
def test_context_slot_rejects_legacy_and_unknown_fields(
    legacy_field: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        ContextSlot(
            name="instructions",
            content="content",
            **legacy_field,
        )


@pytest.mark.parametrize("name", ["", "bad name", "1invalid"])
def test_context_slot_rejects_unsafe_xml_name(name: str) -> None:
    with pytest.raises(ValidationError):
        ContextSlot(name=name, content="content")


@pytest.mark.parametrize("name", ["bad name", "1invalid"])
def test_context_slot_rejects_unsafe_xml_attribute_name(name: str) -> None:
    with pytest.raises(ValidationError):
        ContextSlot(
            name="instructions",
            content="content",
            attributes={name: "value"},
        )


@pytest.mark.parametrize("max_chars", [0, -1])
def test_context_section_rejects_non_positive_max_chars(max_chars: int) -> None:
    with pytest.raises(ValidationError):
        ContextSection(max_chars=max_chars)


def test_context_section_rejects_boolean_max_chars() -> None:
    with pytest.raises(ValidationError):
        ContextSection(max_chars=True)


def test_context_slot_rejects_boolean_order() -> None:
    with pytest.raises(ValidationError):
        ContextSlot(name="instructions", content="content", order=True)


def test_context_integer_fields_accept_regular_integers() -> None:
    slot = ContextSlot(name="instructions", content="content", order=1)
    section = ContextSection(max_chars=1, slots=[slot])

    assert slot.order == 1
    assert section.max_chars == 1


def test_context_section_rejects_relative_template_path() -> None:
    with pytest.raises(ValidationError):
        ContextSection(template=Path("templates/system.xml.j2"))


def test_context_build_input_rejects_empty_system_section() -> None:
    with pytest.raises(ValidationError):
        ContextBuildInput(system=ContextSection())


def test_context_build_input_rejects_only_disabled_system_slots() -> None:
    with pytest.raises(ValidationError):
        ContextBuildInput(
            system=ContextSection(
                slots=[
                    ContextSlot(
                        name="instructions",
                        content="disabled",
                        enabled=False,
                    )
                ]
            )
        )


def test_context_build_input_rejects_legacy_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        ContextBuildInput.model_validate(
            {
                "system": {
                    "slots": [
                        {
                            "name": "instructions",
                            "content": "content",
                        }
                    ]
                },
                "agent_id": "legacy",
            }
        )
