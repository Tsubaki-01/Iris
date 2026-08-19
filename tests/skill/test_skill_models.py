from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

import iris.exceptions as iris_exceptions
from iris.exceptions import (
    IrisError,
    IrisSkillError,
    IrisSkillFormatError,
    IrisSkillNotFoundError,
    IrisSkillPathError,
)
from iris.skill.models import (
    DEFAULT_SKILLS_ROOT,
    MAX_DESCRIPTION_CHARS,
    SKILL_FILE_NAME,
    SkillDiagnostic,
    SkillDiscoveryOptions,
    SkillDiscoveryResult,
    SkillMetadata,
    SkillScope,
)

SKILL_ERROR_TYPES = (
    IrisSkillError,
    IrisSkillFormatError,
    IrisSkillPathError,
    IrisSkillNotFoundError,
)


def _metadata_values(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "name": "example-skill",
        "description": "Example skill",
        "scope": SkillScope.PROJECT,
        "skill_file": Path("C:/workspace/.agents/skills/example-skill/SKILL.md"),
        "root_dir": Path("C:/workspace/.agents/skills/example-skill"),
        "relative_skill_file": ".agents/skills/example-skill/SKILL.md",
        "root_index": 0,
    }
    values.update(overrides)
    return values


def _model_cases() -> tuple[tuple[type[BaseModel], dict[str, Any], str], ...]:
    metadata = SkillMetadata(**_metadata_values())
    diagnostic = SkillDiagnostic(code="name_mismatch", message="Name differs")
    return (
        (SkillMetadata, _metadata_values(), "name"),
        (
            SkillDiagnostic,
            {"code": "name_mismatch", "message": "Name differs"},
            "code",
        ),
        (
            SkillDiscoveryOptions,
            {
                "workspace_root": Path("C:/workspace"),
                "roots": ((SkillScope.PROJECT, Path("C:/workspace/.agents/skills")),),
            },
            "workspace_root",
        ),
        (
            SkillDiscoveryResult,
            {"skills": (metadata,), "diagnostics": (diagnostic,)},
            "skills",
        ),
    )


@pytest.mark.parametrize("error_type", SKILL_ERROR_TYPES)
def test_skill_errors_use_default_runtime_mapping(error_type: type[IrisSkillError]) -> None:
    error = error_type("broken skill", skill="example-skill")

    assert isinstance(error, IrisError)
    assert isinstance(error, ValueError)
    assert error.runtime_source == "runtime"
    assert error.runtime_code == "RUNTIME_ERROR"
    assert error.context == {"skill": "example-skill"}


def test_skill_errors_are_exported_from_exceptions_package() -> None:
    for error_type in SKILL_ERROR_TYPES:
        assert getattr(iris_exceptions, error_type.__name__) is error_type
        assert error_type.__name__ in iris_exceptions.__all__


def test_skill_models_are_frozen() -> None:
    for model_type, values, field_name in _model_cases():
        model = model_type.model_validate(values)

        with pytest.raises(ValidationError, match="frozen"):
            setattr(model, field_name, getattr(model, field_name))


def test_skill_models_forbid_extra_fields() -> None:
    for model_type, values, _ in _model_cases():
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            model_type.model_validate({**values, "unexpected": True})


def test_skill_scope_only_contains_project() -> None:
    assert tuple(SkillScope) == (SkillScope.PROJECT,)
    assert SkillScope.PROJECT.value == "project"


def test_skill_constants_are_stable() -> None:
    assert SKILL_FILE_NAME == "SKILL.md"
    assert DEFAULT_SKILLS_ROOT == ".agents/skills"
    assert MAX_DESCRIPTION_CHARS == 1024


@pytest.mark.parametrize(
    "name",
    ("example", "example-skill", "skill2", "2-skill"),
)
def test_skill_metadata_accepts_lowercase_kebab_case_names(name: str) -> None:
    metadata = SkillMetadata(**_metadata_values(name=name))

    assert metadata.name == name


@pytest.mark.parametrize(
    "name",
    ("Example", "example_skill", "example skill", "-example", "example-", ""),
)
def test_skill_metadata_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValidationError, match="name"):
        SkillMetadata(**_metadata_values(name=name))


def test_skill_metadata_normalizes_and_truncates_description() -> None:
    metadata = SkillMetadata(
        **_metadata_values(description=f"  {'字' * (MAX_DESCRIPTION_CHARS + 1)}  ")
    )

    assert metadata.description == "字" * MAX_DESCRIPTION_CHARS
    assert metadata.description_truncated is True


def test_skill_metadata_preserves_precomputed_truncation() -> None:
    metadata = SkillMetadata(
        **_metadata_values(description="Short", description_truncated=True)
    )

    assert metadata.description == "Short"
    assert metadata.description_truncated is True


def test_skill_metadata_rejects_blank_description() -> None:
    with pytest.raises(ValidationError, match="description"):
        SkillMetadata(**_metadata_values(description="   "))


@pytest.mark.parametrize("max_chars", (True, False, 0, -1, "10"))
def test_discovery_options_require_a_strict_positive_description_limit(
    max_chars: object,
) -> None:
    with pytest.raises(ValidationError, match="max_description_chars"):
        SkillDiscoveryOptions(
            workspace_root=Path("C:/workspace"),
            roots=((SkillScope.PROJECT, Path("C:/workspace/.agents/skills")),),
            max_description_chars=max_chars,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field_name", ("code", "message"))
def test_diagnostic_rejects_blank_required_text(field_name: str) -> None:
    values = {"code": "invalid_skill", "message": "Invalid skill"}
    values[field_name] = "   "

    with pytest.raises(ValidationError, match=field_name):
        SkillDiagnostic(**values)
