from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from fakes import FakeProvider

from iris.agents import (
    AgentConfig,
    AgentContextConfig,
    AgentSkillsConfig,
    ModelConfig,
    PermissionsConfig,
    PythonToolsConfig,
    ToolsConfig,
)
from iris.context import ContextBuilder, ContextBuildInput, ContextSection, ContextSlot
from iris.exceptions import IrisConfigError, IrisContextError
from iris.harness._fingerprint import compute_environment_fingerprint
from iris.message import LLMResponse, TextBlock, ToolUseBlock
from iris.runtime import RuntimeFactory
from iris.runtime.factory import _prepare_skills
from iris.skill import CATALOG_SLOT_NAME, LoadSkillTool
from iris.tools import ReadFileState, ToolExecutionContext


def _provider() -> FakeProvider:
    return FakeProvider(
        [LLMResponse(provider="fake", content=[TextBlock(text="done")])]
    )


def _write_skill(
    workspace: Path,
    name: str,
    *,
    description: str | None = None,
    root_name: str = ".agents/skills",
    body: str = "# Instructions\nFollow this skill.\n",
) -> Path:
    root = workspace / root_name
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.safe_dump(
        {
            "name": name,
            "description": description or f"Use {name}",
        },
        sort_keys=False,
    )
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(f"---\n{frontmatter}---\n{body}", encoding="utf-8")
    return skill_file


def _config(
    workspace: Path,
    *,
    skills: AgentSkillsConfig | None = None,
    tools: ToolsConfig | None = None,
    context: AgentContextConfig | None = None,
) -> AgentConfig:
    values: dict[str, object] = {
        "name": "skill-agent",
        "model": ModelConfig(provider="openai", name="gpt-4o-mini"),
        "permissions": PermissionsConfig(workspace=str(workspace)),
        "tools": tools or ToolsConfig(),
        "skills": skills,
    }
    if context is None:
        values["system"] = "Base instructions"
    else:
        values["context"] = context
    return AgentConfig.model_validate(values)


def _active_tool_names(runtime: object) -> tuple[str, ...]:
    environment = runtime.environment  # type: ignore[attr-defined]
    return tuple(
        tool.definition.name
        for tool in environment.tool_bridge.tool_view.active_tools
    )


def test_prepare_skills_none_is_exact_object_bypass(tmp_path: Path) -> None:
    context_input = ContextBuildInput(
        system=ContextSection(slots=[ContextSlot(name="instructions", content="Base")])
    )

    prepared, registry = _prepare_skills(
        context_input,
        config=_config(tmp_path),
        workspace_root=tmp_path,
    )

    assert prepared is context_input
    assert registry is None


@pytest.mark.parametrize(
    "skills",
    (
        AgentSkillsConfig(enabled=False, root="../outside"),
        AgentSkillsConfig(root="../outside"),
    ),
)
def test_disabled_skills_do_not_resolve_or_scan_root(
    tmp_path: Path,
    skills: AgentSkillsConfig,
) -> None:
    runtime = RuntimeFactory.from_config(
        _config(tmp_path, skills=skills),
        provider=_provider(),
    )

    assert [slot.name for slot in runtime.environment.context_input.system.slots] == [
        "instructions"
    ]
    assert "load_skill" not in _active_tool_names(runtime)


def test_enabled_missing_root_logs_warning_without_slot_or_loader(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="iris.runtime.factory"):
        runtime = RuntimeFactory.from_config(
            _config(tmp_path, skills=AgentSkillsConfig(enabled=True)),
            provider=_provider(),
        )

    assert [slot.name for slot in runtime.environment.context_input.system.slots] == [
        "instructions"
    ]
    assert "load_skill" not in _active_tool_names(runtime)
    assert any(getattr(record, "code", None) == "ROOT_MISSING" for record in caplog.records)


def test_required_missing_is_config_error_with_stable_context(tmp_path: Path) -> None:
    _write_skill(tmp_path, "available-skill")

    with pytest.raises(IrisConfigError, match="required") as exc_info:
        RuntimeFactory.from_config(
            _config(
                tmp_path,
                skills=AgentSkillsConfig(
                    enabled=True,
                    require=("missing-skill",),
                ),
            ),
            provider=_provider(),
        )

    assert exc_info.value.context == {
        "missing": ("missing-skill",),
        "available": ("available-skill",),
    }


def test_required_existing_builds_successfully(tmp_path: Path) -> None:
    _write_skill(tmp_path, "required-skill")

    runtime = RuntimeFactory.from_config(
        _config(
            tmp_path,
            skills=AgentSkillsConfig(enabled=True, require=("required-skill",)),
        ),
        provider=_provider(),
    )

    assert "load_skill" in _active_tool_names(runtime)


def test_root_outside_workspace_is_converted_to_config_error(tmp_path: Path) -> None:
    with pytest.raises(IrisConfigError, match="skills.root") as exc_info:
        RuntimeFactory.from_config(
            _config(
                tmp_path,
                skills=AgentSkillsConfig(enabled=True, root="../outside"),
            ),
            provider=_provider(),
        )

    assert exc_info.value.context["root"] == "../outside"
    assert exc_info.value.context["workspace_root"] == str(tmp_path.resolve())


@pytest.mark.asyncio
async def test_catalog_and_loader_share_snapshot_and_execute_without_file_builtin(
    tmp_path: Path,
) -> None:
    skill_file = _write_skill(tmp_path, "example-skill")
    runtime = RuntimeFactory.from_config(
        _config(tmp_path, skills=AgentSkillsConfig(enabled=True)),
        provider=_provider(),
    )
    bridge = runtime.environment.tool_bridge
    view_tool = bridge.tool_view.get("load_skill")
    executor_tool = bridge.tool_executor.registry.get("load_skill")

    assert isinstance(view_tool, LoadSkillTool)
    assert view_tool is executor_tool
    assert "read_file" not in _active_tool_names(runtime)

    context = ToolExecutionContext(workspace_root=tmp_path)
    loaded = await bridge.tool_executor.execute_one(
        ToolUseBlock(
            id="call-load",
            name="load_skill",
            input={"name": "example-skill"},
        ),
        context,
    )
    assert loaded.is_error is False
    assert "# Instructions" in loaded.model_content
    assert isinstance(context.read_state, ReadFileState)
    assert context.read_state.get(skill_file.resolve()) is not None


def test_catalog_renders_after_default_order_user_slots(tmp_path: Path) -> None:
    _write_skill(tmp_path, "example-skill")
    runtime = RuntimeFactory.from_config(
        _config(tmp_path, skills=AgentSkillsConfig(enabled=True)),
        provider=_provider(),
    )

    rendered = runtime.environment.context_builder.build(
        runtime.environment.context_input
    ).system.text

    assert rendered.index("<instructions>") < rendered.index(
        f"<{CATALOG_SLOT_NAME}"
    )


def test_custom_template_warns_and_build_succeeds(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_skill(tmp_path, "example-skill")
    template = tmp_path / "system.xml.j2"
    template.write_text(
        "<custom>{% for slot in slots %}<slot>{{ slot.name }}</slot>{% endfor %}</custom>",
        encoding="utf-8",
    )
    context_path = tmp_path / "context.yaml"
    context_path.write_text(
        "\n".join(
            [
                "system:",
                "  template: system.xml.j2",
                "  slots:",
                "    - name: instructions",
                "      content: Base",
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="iris.runtime.factory"):
        runtime = RuntimeFactory.from_config(
            _config(
                tmp_path,
                skills=AgentSkillsConfig(enabled=True),
                context=AgentContextConfig(path=context_path.resolve()),
            ),
            provider=_provider(),
        )

    assert any(getattr(record, "code", None) == "TEMPLATE_SECTION" for record in caplog.records)
    rendered = runtime.environment.context_builder.build(
        runtime.environment.context_input
    ).system.text
    assert "available_skills" in rendered


def test_same_tree_has_stable_fingerprint_and_description_change_changes_it(
    tmp_path: Path,
) -> None:
    skill_file = _write_skill(tmp_path, "example-skill", description="First")
    config = _config(tmp_path, skills=AgentSkillsConfig(enabled=True))

    first = RuntimeFactory.from_config(config, provider=_provider())
    second = RuntimeFactory.from_config(config, provider=_provider())
    first_fingerprint = compute_environment_fingerprint(first)

    assert compute_environment_fingerprint(second) == first_fingerprint

    text = skill_file.read_text(encoding="utf-8").replace(
        "description: First",
        "description: Changed",
    )
    skill_file.write_text(text, encoding="utf-8")
    changed = RuntimeFactory.from_config(config, provider=_provider())

    assert compute_environment_fingerprint(changed) != first_fingerprint


def test_user_python_tool_named_load_skill_is_config_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_skill(tmp_path, "example-skill")
    module_path = tmp_path / "conflict_tools.py"
    module_path.write_text(
        "def load_skill(name: str) -> str:\n"
        "    \"\"\"Conflicting user tool.\"\"\"\n"
        "    return name\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(IrisConfigError, match="load_skill") as exc_info:
        RuntimeFactory.from_config(
            _config(
                tmp_path,
                skills=AgentSkillsConfig(enabled=True),
                tools=ToolsConfig(
                    python=PythonToolsConfig(
                        functions=["conflict_tools:load_skill"]
                    )
                ),
            ),
            provider=_provider(),
        )

    assert exc_info.value.context["tool"] == "load_skill"


def test_user_tool_alias_named_load_skill_is_config_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_skill(tmp_path, "example-skill")
    module_path = tmp_path / "alias_tools.py"
    module_path.write_text(
        "from iris.tools import BaseTool, ToolDefinition, ToolResult\n\n"
        "class AliasTool(BaseTool):\n"
        "    definition = ToolDefinition(\n"
        "        name='user_loader',\n"
        "        description='Conflicting alias tool',\n"
        "        input_schema={'type': 'object'},\n"
        "        aliases=('load_skill',),\n"
        "    )\n\n"
        "    async def arun(self, params, context):\n"
        "        return ToolResult(tool_use_id='', tool_name='user_loader')\n\n"
        "def register_tools(registry):\n"
        "    registry.register(AliasTool())\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(IrisConfigError, match="load_skill") as exc_info:
        RuntimeFactory.from_config(
            _config(
                tmp_path,
                skills=AgentSkillsConfig(enabled=True),
                tools=ToolsConfig(
                    python=PythonToolsConfig(
                        registrars=["alias_tools:register_tools"]
                    )
                ),
            ),
            provider=_provider(),
        )

    assert exc_info.value.context["tool"] == "load_skill"


def test_explicit_small_system_budget_fails_without_catalog_omission(
    tmp_path: Path,
) -> None:
    for index in range(5):
        _write_skill(tmp_path, f"skill-{index}")
    context_input = ContextBuildInput(
        system=ContextSection(
            max_chars=50,
            slots=[ContextSlot(name="instructions", content="Base")],
        )
    )
    prepared, registry = _prepare_skills(
        context_input,
        config=_config(tmp_path, skills=AgentSkillsConfig(enabled=True)),
        workspace_root=tmp_path,
    )

    assert registry is not None and len(registry) == 5
    catalog_slot = next(
        slot for slot in prepared.system.slots if slot.name == CATALOG_SLOT_NAME
    )
    assert len(catalog_slot.content) == 5
    with pytest.raises(IrisContextError) as exc_info:
        ContextBuilder().build(prepared)
    assert exc_info.value.context["section"] == "system"
    assert exc_info.value.context["actual"] > exc_info.value.context["limit"]


def test_unbounded_system_keeps_all_skills_and_logs_exact_metrics(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for index in range(12):
        _write_skill(tmp_path, f"skill-{index}")
    context_input = ContextBuildInput(
        system=ContextSection(
            slots=[ContextSlot(name="instructions", content="Base")]
        )
    )

    with caplog.at_level(logging.INFO, logger="iris.runtime.factory"):
        prepared, registry = _prepare_skills(
            context_input,
            config=_config(tmp_path, skills=AgentSkillsConfig(enabled=True)),
            workspace_root=tmp_path,
        )

    assert registry is not None and len(registry) == 12
    output = ContextBuilder().build(prepared)
    assert output.system.text.count('name="name"') == 12
    info_record = next(
        record for record in caplog.records if record.levelno == logging.INFO
    )
    assert info_record.count == 12
    assert info_record.content_chars > 0
