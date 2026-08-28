from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from iris.exceptions import IrisSkillFormatError, IrisSkillPathError
from iris.skill.discovery import _load_skill, discover_skills, resolve_skills_root
from iris.skill.models import (
    MAX_DESCRIPTION_CHARS,
    SkillDiagnostic,
    SkillDiscoveryOptions,
    SkillDiscoveryResult,
    SkillScope,
)


def _write_skill(
    root: Path,
    name: str,
    *,
    frontmatter: dict[str, Any] | None = None,
    body: str = "# Skill body\n",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    values = {"description": f"Use {name}"} if frontmatter is None else frontmatter
    yaml_text = yaml.safe_dump(values, sort_keys=False, allow_unicode=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(f"---\n{yaml_text}---\n{body}", encoding="utf-8")
    return skill_file


def _discover(
    workspace: Path,
    *roots: Path,
    max_description_chars: int = MAX_DESCRIPTION_CHARS,
) -> SkillDiscoveryResult:
    return discover_skills(
        SkillDiscoveryOptions(
            workspace_root=workspace,
            roots=tuple((SkillScope.PROJECT, root) for root in roots),
            max_description_chars=max_description_chars,
        )
    )


def _diagnostics(result: SkillDiscoveryResult, code: str) -> list[SkillDiagnostic]:
    return [diagnostic for diagnostic in result.diagnostics if diagnostic.code == code]


def _symlink_or_skip(link: Path, target: Path, *, is_directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=is_directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"当前 Windows 环境无法创建符号链接: {exc}")


def test_discovers_three_valid_skills_without_diagnostics(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / ".agents" / "skills"
    for name in ("gamma", "alpha", "beta"):
        _write_skill(root, name)

    result = _discover(workspace, root)

    assert tuple(skill.name for skill in result.skills) == ("alpha", "beta", "gamma")
    assert result.diagnostics == ()
    assert all(skill.root_index == 0 for skill in result.skills)
    assert all(skill.skill_file.is_absolute() for skill in result.skills)
    assert all(not hasattr(skill, "body") for skill in result.skills)


def test_missing_root_returns_diagnostic_without_raising(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    missing_root = workspace / ".agents" / "skills"

    result = _discover(workspace, missing_root)

    assert result.skills == ()
    assert [item.code for item in result.diagnostics] == ["ROOT_MISSING"]
    assert result.diagnostics[0].path == missing_root.resolve()
    assert result.diagnostics[0].detail == {"scope": "project", "root_index": "0"}


def test_root_that_is_a_file_returns_root_missing_diagnostic(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = workspace / "skills"
    root.write_text("not a directory", encoding="utf-8")

    result = _discover(workspace, root)

    assert result.skills == ()
    assert [item.code for item in result.diagnostics] == ["ROOT_MISSING"]


def test_missing_skill_file_does_not_block_valid_siblings(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    (root / "missing-file").mkdir(parents=True)
    _write_skill(root, "valid-skill")

    result = _discover(workspace, root)

    assert tuple(skill.name for skill in result.skills) == ("valid-skill",)
    assert [item.code for item in result.diagnostics] == ["MISSING_SKILL_FILE"]
    assert result.diagnostics[0].path == (root / "missing-file").resolve()


def test_skill_file_name_is_case_sensitive(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    skill_dir = root / "wrong-case"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.md").write_text("---\ndescription: wrong\n---\n", encoding="utf-8")

    result = _discover(workspace, root)

    assert result.skills == ()
    assert [item.code for item in result.diagnostics] == ["MISSING_SKILL_FILE"]


def test_invalid_yaml_does_not_block_valid_siblings(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    _write_skill(root, "alpha")
    bad_file = _write_skill(root, "broken")
    bad_file.write_text("---\ndescription: [unterminated\n---\n", encoding="utf-8")
    _write_skill(root, "charlie")

    result = _discover(workspace, root)

    assert tuple(skill.name for skill in result.skills) == ("alpha", "charlie")
    invalid = _diagnostics(result, "INVALID_SKILL")
    assert len(invalid) == 1
    assert invalid[0].path == bad_file.resolve()
    assert "YAML 解析失败" in invalid[0].detail["reason"]


@pytest.mark.parametrize("name", ("Bad_Name", "bad name"))
def test_invalid_directory_name_is_diagnosed(tmp_path: Path, name: str) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    _write_skill(root, name)

    result = _discover(workspace, root)

    assert result.skills == ()
    assert [item.code for item in result.diagnostics] == ["INVALID_NAME"]
    assert result.diagnostics[0].detail == {"name": name}


def test_declared_name_mismatch_keeps_directory_key(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    skill_file = _write_skill(
        root,
        "directory-name",
        frontmatter={"name": "declared-name", "description": "Example"},
    )

    result = _discover(workspace, root)

    assert tuple(skill.name for skill in result.skills) == ("directory-name",)
    assert result.skills[0].declared_name == "declared-name"
    mismatch = _diagnostics(result, "NAME_MISMATCH")
    assert len(mismatch) == 1
    assert mismatch[0].path == skill_file.resolve()
    assert mismatch[0].detail == {
        "declared_name": "declared-name",
        "directory_name": "directory-name",
    }


def test_hidden_directories_are_skipped_silently(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    _write_skill(root, ".hidden")
    _write_skill(root, "visible")

    result = _discover(workspace, root)

    assert tuple(skill.name for skill in result.skills) == ("visible",)
    assert result.diagnostics == ()


def test_discovery_does_not_recurse_below_first_level(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    _write_skill(root / "container", "nested-skill")

    result = _discover(workspace, root)

    assert result.skills == ()
    assert [item.code for item in result.diagnostics] == ["MISSING_SKILL_FILE"]


@pytest.mark.parametrize(
    "frontmatter",
    ({}, {"description": "   "}, {"description": ["not", "text"]}),
)
def test_invalid_description_rejects_only_that_skill(
    tmp_path: Path,
    frontmatter: dict[str, Any],
) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    _write_skill(root, "broken", frontmatter=frontmatter)
    _write_skill(root, "valid")

    result = _discover(workspace, root)

    assert tuple(skill.name for skill in result.skills) == ("valid",)
    invalid = _diagnostics(result, "INVALID_SKILL")
    assert len(invalid) == 1
    assert "description" in invalid[0].detail["reason"]


def test_unknown_frontmatter_fields_are_preserved_without_dedicated_attributes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    _write_skill(
        root,
        "metadata-skill",
        frontmatter={
            "description": "Example",
            "allowed-tools": ["read_file"],
            "disable-model-invocation": True,
            "version": 2,
        },
    )

    metadata = _discover(workspace, root).skills[0]

    assert metadata.extra_frontmatter == {
        "allowed-tools": ["read_file"],
        "disable-model-invocation": True,
        "version": 2,
    }
    assert not hasattr(metadata, "allowed_tools")
    assert not hasattr(metadata, "disable_model_invocation")


def test_long_description_is_truncated_and_diagnosed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    skill_file = _write_skill(
        root,
        "long-description",
        frontmatter={"description": "字" * 2000},
    )

    result = _discover(workspace, root)

    assert result.skills[0].description == "字" * MAX_DESCRIPTION_CHARS
    assert result.skills[0].description_truncated is True
    truncated = _diagnostics(result, "DESCRIPTION_TRUNCATED")
    assert len(truncated) == 1
    assert truncated[0].path == skill_file.resolve()
    assert truncated[0].detail == {"limit": str(MAX_DESCRIPTION_CHARS)}


def test_custom_description_limit_is_applied(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    _write_skill(root, "custom-limit", frontmatter={"description": "1234567890"})

    result = _discover(workspace, root, max_description_chars=5)

    assert result.skills[0].description == "12345"
    assert result.skills[0].description_truncated is True
    assert _diagnostics(result, "DESCRIPTION_TRUNCATED")[0].detail == {"limit": "5"}


def test_first_root_wins_duplicate_name_with_complete_diagnostic(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    first_root = workspace / "first"
    second_root = workspace / "second"
    _write_skill(first_root, "shared", frontmatter={"description": "First"})
    _write_skill(second_root, "shared", frontmatter={"description": "Second"})

    result = _discover(workspace, first_root, second_root)

    assert len(result.skills) == 1
    assert result.skills[0].description == "First"
    assert result.skills[0].root_index == 0
    duplicate = _diagnostics(result, "DUPLICATE_NAME")
    assert len(duplicate) == 1
    assert duplicate[0].path == (second_root / "shared").resolve()
    assert duplicate[0].detail == {
        "winner_path": str((first_root / "shared").resolve()),
        "winner_scope": "project",
        "loser_path": str((second_root / "shared").resolve()),
        "loser_scope": "project",
    }
    assert all(
        Path(duplicate[0].detail[key]).is_absolute()
        for key in ("winner_path", "loser_path")
    )


def test_same_real_root_is_silently_deduplicated(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    _write_skill(root, "shared")

    result = _discover(workspace, root, root)

    assert tuple(skill.name for skill in result.skills) == ("shared",)
    assert _diagnostics(result, "DUPLICATE_NAME") == []


def test_symlinked_root_alias_is_silently_deduplicated(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    _write_skill(root, "shared")
    alias = workspace / "skills-alias"
    _symlink_or_skip(alias, root, is_directory=True)

    result = _discover(workspace, root, alias)

    assert tuple(skill.name for skill in result.skills) == ("shared",)
    assert _diagnostics(result, "DUPLICATE_NAME") == []


def test_relative_root_is_resolved_from_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    resolved = resolve_skills_root(".agents/skills", workspace_root=workspace)

    assert resolved == (workspace / ".agents" / "skills").resolve()


def test_blank_root_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(IrisSkillPathError, match="不能为空"):
        resolve_skills_root("   ", workspace_root=workspace)


def test_absolute_root_outside_workspace_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"

    with pytest.raises(IrisSkillPathError, match="workspace") as exc_info:
        resolve_skills_root(str(outside.resolve()), workspace_root=workspace)

    assert exc_info.value.context["path"] == str(outside.resolve())
    assert exc_info.value.context["workspace_root"] == str(workspace.resolve())


def test_parent_traversal_outside_workspace_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(IrisSkillPathError, match="workspace"):
        resolve_skills_root("../outside", workspace_root=workspace)


def test_discovery_defensively_rejects_outside_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(IrisSkillPathError, match="workspace"):
        _discover(workspace, tmp_path / "outside")


def test_skill_directory_symlink_escape_is_invalid_but_siblings_survive(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    root.mkdir(parents=True)
    outside_dir = tmp_path / "outside-skill"
    _write_skill(tmp_path, "outside-skill")
    _symlink_or_skip(root / "escaped-skill", outside_dir, is_directory=True)
    _write_skill(root, "valid")

    result = _discover(workspace, root)

    assert tuple(skill.name for skill in result.skills) == ("valid",)
    assert [item.code for item in result.diagnostics] == ["INVALID_SKILL"]


def test_escaped_skill_directory_without_skill_file_is_invalid_not_missing(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    root.mkdir(parents=True)
    outside_dir = tmp_path / "outside-empty"
    outside_dir.mkdir()
    _symlink_or_skip(root / "escaped-skill", outside_dir, is_directory=True)

    result = _discover(workspace, root)

    assert result.skills == ()
    assert [item.code for item in result.diagnostics] == ["INVALID_SKILL"]


def test_skill_file_symlink_escape_is_invalid_but_siblings_survive(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    escaped_dir = root / "escaped-file"
    escaped_dir.mkdir(parents=True)
    outside_file = tmp_path / "outside.md"
    outside_file.write_text("---\ndescription: Outside\n---\n", encoding="utf-8")
    _symlink_or_skip(escaped_dir / "SKILL.md", outside_file, is_directory=False)
    _write_skill(root, "valid")

    result = _discover(workspace, root)

    assert tuple(skill.name for skill in result.skills) == ("valid",)
    assert [item.code for item in result.diagnostics] == ["INVALID_SKILL"]


def test_root_internal_directory_symlink_is_allowed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    target_file = _write_skill(root, ".target")
    _symlink_or_skip(root / "linked-skill", target_file.parent, is_directory=True)

    result = _discover(workspace, root)

    assert tuple(skill.name for skill in result.skills) == ("linked-skill",)
    assert result.skills[0].root_dir == target_file.parent.resolve()
    assert result.diagnostics == ()


def test_discovery_is_stable_across_creation_order_and_repeated_calls(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    for name in ("charlie", "alpha", "bravo"):
        _write_skill(root, name)
    first = _discover(workspace, root)

    shutil.rmtree(root)
    for name in ("bravo", "charlie", "alpha"):
        _write_skill(root, name)
    second = _discover(workspace, root)
    third = _discover(workspace, root)

    assert first == second == third
    assert tuple(skill.name for skill in first.skills) == ("alpha", "bravo", "charlie")


def test_final_order_is_root_priority_then_name(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    first_root = workspace / "first"
    second_root = workspace / "second"
    for name in ("delta", "bravo"):
        _write_skill(first_root, name)
    for name in ("charlie", "alpha"):
        _write_skill(second_root, name)

    result = _discover(workspace, first_root, second_root)

    assert tuple(skill.name for skill in result.skills) == (
        "bravo",
        "delta",
        "alpha",
        "charlie",
    )
    assert tuple(skill.root_index for skill in result.skills) == (0, 0, 1, 1)


def test_parser_error_receives_absolute_file_path_at_load_boundary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    skill_file = _write_skill(root, "broken")
    skill_file.write_text("---\ndescription: missing close\n", encoding="utf-8")

    with pytest.raises(IrisSkillFormatError) as exc_info:
        _load_skill(
            skill_file.parent,
            workspace_root=workspace,
            scope=SkillScope.PROJECT,
            root=root,
            root_index=0,
            max_description_chars=MAX_DESCRIPTION_CHARS,
        )

    assert exc_info.value.context["path"] == str(skill_file.resolve())
    assert isinstance(exc_info.value.__cause__, IrisSkillFormatError)


def test_non_utf8_skill_is_invalid_without_blocking_siblings(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "skills"
    bad_file = _write_skill(root, "binary-skill")
    bad_file.write_bytes(b"\xff\xfe\x00")
    _write_skill(root, "valid")

    result = _discover(workspace, root)

    assert tuple(skill.name for skill in result.skills) == ("valid",)
    invalid = _diagnostics(result, "INVALID_SKILL")
    assert len(invalid) == 1
    assert "UTF-8" in invalid[0].detail["reason"]
