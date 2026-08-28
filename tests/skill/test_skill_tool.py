from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from iris.exceptions import IrisSkillPathError
from iris.message import ToolUseBlock
from iris.skill.discovery import discover_skills
from iris.skill.models import SkillDiscoveryOptions, SkillScope
from iris.skill.registry import SkillRegistry
from iris.skill.tool import LoadSkillInput, LoadSkillTool
from iris.tools import (
    PermissionEffect,
    ReadFileState,
    ToolCapability,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
)


def _write_skill(
    workspace: Path,
    *,
    name: str = "example-skill",
    body: str = "# Instructions\nFollow the skill.\n",
) -> tuple[Path, Path]:
    root = workspace / ".agents" / "skills"
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.safe_dump(
        {"name": name, "description": f"Use {name}"},
        sort_keys=False,
    )
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(f"---\n{frontmatter}---\n{body}", encoding="utf-8")
    return root, skill_file


def _registry(workspace: Path, **skill_kwargs: Any) -> tuple[SkillRegistry, Path]:
    root, skill_file = _write_skill(workspace, **skill_kwargs)
    result = discover_skills(
        SkillDiscoveryOptions(
            workspace_root=workspace,
            roots=((SkillScope.PROJECT, root),),
        )
    )
    return SkillRegistry(result), skill_file


def _executor(tool: LoadSkillTool) -> ToolExecutor:
    tools = ToolRegistry()
    tools.register(tool)
    return ToolExecutor(tools)


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"当前 Windows 环境无法创建符号链接: {exc}")


def test_load_skill_definition_and_input_schema_are_exact(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    tool = LoadSkillTool(registry, max_result_chars=1234)

    assert tool.definition.name == "load_skill"
    assert tool.definition.group == "skill"
    assert tool.definition.capabilities == {ToolCapability.READ}
    assert tool.definition.deferred is False
    assert tool.definition.max_result_chars == 1234
    assert set(tool.definition.input_schema["properties"]) == {"name"}
    assert tool.definition.input_schema["required"] == ["name"]
    assert "不执行脚本" in tool.definition.description
    assert tool.input_model is LoadSkillInput


def test_load_skill_input_rejects_invalid_names_and_extra_paths() -> None:
    with pytest.raises(ValidationError):
        LoadSkillInput(name="Bad_Name")
    with pytest.raises(ValidationError):
        LoadSkillInput.model_validate({"name": "example-skill", "path": "secret"})


@pytest.mark.asyncio
async def test_valid_skill_returns_markdown_and_records_real_read_path(
    tmp_path: Path,
) -> None:
    registry, skill_file = _registry(tmp_path)
    tool = LoadSkillTool(registry)
    context = ToolExecutionContext(workspace_root=tmp_path)

    result = await tool.arun({"name": "example-skill"}, context)

    expected = "\n".join(skill_file.read_text(encoding="utf-8").splitlines()[:1000])
    assert result.is_error is False
    assert result.tool_use_id == ""
    assert result.tool_name == "load_skill"
    assert result.model_content == expected
    assert isinstance(context.read_state, ReadFileState)
    assert context.read_state.get(skill_file.resolve()) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_field", ("path", "root", "scope"))
async def test_executor_owns_input_validation_errors(
    tmp_path: Path,
    extra_field: str,
) -> None:
    registry, _ = _registry(tmp_path)
    result = await _executor(LoadSkillTool(registry)).execute_one(
        ToolUseBlock(
            id="call-validation",
            name="load_skill",
            input={"name": "example-skill", extra_field: "outside"},
        ),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_missing_registry_name_returns_retryable_error(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)

    result = await LoadSkillTool(registry).arun(
        {"name": "missing-skill"},
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "SKILL_NOT_FOUND"
    assert result.error.retryable is True
    assert result.error.details == {
        "name": "missing-skill",
        "available": ["example-skill"],
    }


@pytest.mark.asyncio
async def test_post_discovery_symlink_escape_returns_path_error_without_secret(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry, skill_file = _registry(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-secret.md"
    outside.write_text("TOP SECRET", encoding="utf-8")
    skill_file.unlink()
    _symlink_or_skip(skill_file, outside)

    with caplog.at_level(logging.WARNING, logger="iris.skill.tool"):
        result = await LoadSkillTool(registry).arun(
            {"name": "example-skill"},
            ToolExecutionContext(workspace_root=tmp_path),
        )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "SKILL_PATH_ERROR"
    assert result.error.retryable is False
    assert result.error.details == {"name": "example-skill"}
    assert "TOP SECRET" not in result.model_content
    logged_error = caplog.records[0].args[1]
    assert isinstance(logged_error, IrisSkillPathError)
    assert logged_error.context["path"] == str(outside.resolve())


@pytest.mark.asyncio
async def test_post_discovery_sibling_retarget_returns_path_error_without_secret(
    tmp_path: Path,
) -> None:
    registry, skill_file = _registry(tmp_path)
    sibling_dir = skill_file.parent.parent / "sibling-skill"
    sibling_dir.mkdir()
    sibling_file = sibling_dir / "SKILL.md"
    sibling_file.write_text("SIBLING SECRET", encoding="utf-8")
    skill_file.unlink()
    _symlink_or_skip(skill_file, sibling_file)

    result = await LoadSkillTool(registry).arun(
        {"name": "example-skill"},
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "SKILL_PATH_ERROR"
    assert "SIBLING SECRET" not in result.model_content


@pytest.mark.asyncio
async def test_deleted_skill_file_returns_retryable_read_error(tmp_path: Path) -> None:
    registry, skill_file = _registry(tmp_path)
    skill_file.unlink()

    result = await LoadSkillTool(registry).arun(
        {"name": "example-skill"},
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "SKILL_READ_ERROR"
    assert result.error.retryable is True
    assert result.error.details == {
        "name": "example-skill",
        "reason": "SKILL.md 不存在或不是普通文件",
    }


@pytest.mark.asyncio
async def test_non_utf8_skill_returns_retryable_read_error(tmp_path: Path) -> None:
    registry, skill_file = _registry(tmp_path)
    skill_file.write_bytes(b"\xff\xfe\x00")

    result = await LoadSkillTool(registry).arun(
        {"name": "example-skill"},
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "SKILL_READ_ERROR"
    assert result.error.details["reason"] == "SKILL.md 不是有效的 UTF-8 文本"


@pytest.mark.asyncio
async def test_loader_preserves_file_service_thousand_line_limit(tmp_path: Path) -> None:
    body = "".join(f"line-{index}\n" for index in range(1, 1002))
    registry, _ = _registry(tmp_path, body=body)

    result = await LoadSkillTool(registry).arun(
        {"name": "example-skill"},
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is False
    assert len(result.model_content.splitlines()) == 1000
    assert "line-1001" not in result.model_content


@pytest.mark.asyncio
async def test_executor_allows_read_without_human_gate(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    executor = _executor(LoadSkillTool(registry))
    context = ToolExecutionContext(workspace_root=tmp_path)
    call = ToolUseBlock(id="call-load", name="load_skill", input={"name": "example-skill"})

    plan = executor.prepare_many([call], context)
    result = await executor.execute_one(call, context)

    assert plan.calls[0].permission is not None
    assert plan.calls[0].permission.effect is PermissionEffect.ALLOW
    assert plan.calls[0].human_request is None
    assert result.is_error is False
    assert result.tool_use_id == "call-load"
    assert result.artifact is None
