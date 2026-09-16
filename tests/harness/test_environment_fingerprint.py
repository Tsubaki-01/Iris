"""恢复指纹绑定有效运行语义，不绑定配置文件的摆放方式。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import iris.config as config_module
from iris.agents import (
    AgentConfig,
    AgentContextConfig,
    CompactionConfig,
    ModelConfig,
    SessionConfig,
)
from iris.config import Config, ProviderConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.exceptions import IrisContextError
from iris.harness._fingerprint import compute_environment_fingerprint
from iris.runtime import AgentRuntime, RuntimeFactory

from .fakes import StaticProvider, build_runtime, text_response


def _templated_runtime(workspace: Path, template: Path) -> AgentRuntime:
    """创建只有模板来源不同的相同运行环境。"""
    runtime = build_runtime(workspace)
    runtime.environment.context_input = ContextBuildInput(
        system=ContextSection(
            template=template,
            slots=[ContextSlot(name="instructions", content="hello")],
        )
    )
    return runtime


def test_fingerprint_ignores_session_storage_configuration(tmp_path: Path) -> None:
    """会话存储位置不改变同一运行的模型、提示或工具行为。"""
    first = build_runtime(tmp_path)
    second = build_runtime(tmp_path)
    second.environment.agent_config = second.environment.agent_config.model_copy(
        update={"session": SessionConfig(backend="sqlite", path="another/session.db")}
    )
    assert compute_environment_fingerprint(first) == compute_environment_fingerprint(second)


@pytest.mark.parametrize(
    "override",
    [
        {"input_budget_tokens": 32000},
        {"keep_recent_ratio": 0.2},
        {"summary_ratio": 0.1},
        {"timeout_seconds": 60},
    ],
)
def test_fingerprint_binds_compaction_configuration(
    tmp_path: Path, override: dict[str, float]
) -> None:
    """压缩配置影响运行语义，纳入既有恢复指纹。"""
    first = build_runtime(tmp_path)
    second = build_runtime(tmp_path)
    second.environment.agent_config = second.environment.agent_config.model_copy(
        update={"compaction": CompactionConfig.model_validate(override)}
    )

    assert compute_environment_fingerprint(first) != compute_environment_fingerprint(second)


def test_fingerprint_ignores_template_location_but_binds_content(tmp_path: Path) -> None:
    """移动相同模板不误拒绝恢复，修改正文会改变恢复契约。"""
    first_path = tmp_path / "first.j2"
    second_path = tmp_path / "second.j2"
    first_path.write_text("<system>{{ slots[0].content }}</system>", encoding="utf-8")
    second_path.write_text(first_path.read_text(encoding="utf-8"), encoding="utf-8")
    first = compute_environment_fingerprint(_templated_runtime(tmp_path, first_path))
    same = compute_environment_fingerprint(_templated_runtime(tmp_path, second_path))
    assert first == same

    first_path.write_text("<system>new instructions</system>", encoding="utf-8")
    changed = compute_environment_fingerprint(_templated_runtime(tmp_path, first_path))
    assert changed != first


def test_fingerprint_binds_included_template_and_keeps_runtime_snapshot(tmp_path: Path) -> None:
    """已运行实例使用原模板快照，新实例识别实际 include 的内容变化。"""
    template = tmp_path / "main.j2"
    included = tmp_path / "instructions.j2"
    template.write_text('{% include "instructions.j2" %}', encoding="utf-8")
    included.write_text("original instructions", encoding="utf-8")
    existing = _templated_runtime(tmp_path, template)
    original = compute_environment_fingerprint(existing)
    included.write_text("updated instructions", encoding="utf-8")
    assert compute_environment_fingerprint(existing) == original
    assert (
        existing.environment.context_builder.build(existing.environment.context_input).system.text
        == "original instructions"
    )
    assert compute_environment_fingerprint(_templated_runtime(tmp_path, template)) != original


def test_fingerprint_ignores_disabled_context_and_empty_before_input_template(
    tmp_path: Path,
) -> None:
    """没有进入渲染的 slot 和空前置段不影响恢复，也不读取其模板。"""
    first = build_runtime(tmp_path)
    second = build_runtime(tmp_path)
    second.environment.context_input.system.slots.append(
        ContextSlot(name="disabled", content=object(), enabled=False)
    )
    second.environment.context_input.before_current_input = ContextSection(
        template=tmp_path / "missing.j2",
        max_chars=1,
        slots=[ContextSlot(name="disabled", content="unused", enabled=False)],
    )
    assert compute_environment_fingerprint(first) == compute_environment_fingerprint(second)


def test_fingerprint_binds_memory_template_before_dynamic_slots_arrive(tmp_path: Path) -> None:
    """空 memory 后续可由 run options 激活，模板版本在启动时已冻结。"""
    template = tmp_path / "memory.j2"
    template.write_text("original {{ slots[0].content }}", encoding="utf-8")
    existing = build_runtime(tmp_path)
    existing.environment.context_input.memory = ContextSection(template=template)
    original = compute_environment_fingerprint(existing)
    template.write_text("changed {{ slots[0].content }}", encoding="utf-8")
    with_memory = existing.environment.context_input.with_memory_slots(
        ContextSlot(name="memory", content="recalled")
    )
    output = existing.environment.context_builder.build(with_memory)
    assert output.memory is not None
    assert output.memory.text == "original recalled"
    changed = build_runtime(tmp_path)
    changed.environment.context_input.memory = ContextSection(template=template)
    assert compute_environment_fingerprint(changed) != original


@pytest.mark.parametrize("template_text,max_chars", [("{{ missing }}", None), ("hello", 1)])
def test_fingerprint_does_not_render_context(
    tmp_path: Path,
    template_text: str,
    max_chars: int | None,
) -> None:
    """指纹读取模板来源，StrictUndefined 与字符上限仍在实际渲染时校验。"""
    template = tmp_path / "main.j2"
    template.write_text(template_text, encoding="utf-8")
    runtime = _templated_runtime(tmp_path, template)
    runtime.environment.context_input.system.max_chars = max_chars
    assert compute_environment_fingerprint(runtime)
    with pytest.raises(IrisContextError):
        runtime.environment.context_builder.build(runtime.environment.context_input)


def test_fingerprint_preserves_actionable_template_source_error(tmp_path: Path) -> None:
    """来源契约错误保持 context 归属，不被误报为 JSON 编码失败。"""
    template = tmp_path / "main.j2"
    template.write_text("{% include selected %}", encoding="utf-8")
    with pytest.raises(IrisContextError, match="静态文件名"):
        compute_environment_fingerprint(_templated_runtime(tmp_path, template))


def test_fingerprint_uses_loaded_context_instead_of_prompt_declaration(tmp_path: Path) -> None:
    """相同已加载 context 不因 system/context 声明来源不同而改变恢复契约。"""
    first = build_runtime(tmp_path)
    second = build_runtime(tmp_path)
    second.environment.agent_config = second.environment.agent_config.model_copy(
        update={"system": None, "context": AgentContextConfig(path=tmp_path / "context.yaml")}
    )
    assert compute_environment_fingerprint(first) == compute_environment_fingerprint(second)


def test_fingerprint_ignores_default_empty_memory_section(tmp_path: Path) -> None:
    """未配置模板或上限的空 memory 与缺省段具有相同后续注入行为。"""
    first = build_runtime(tmp_path)
    second = build_runtime(tmp_path)
    second.environment.context_input.memory = ContextSection()
    assert compute_environment_fingerprint(first) == compute_environment_fingerprint(second)


@pytest.mark.parametrize("change", ["temperature", "workspace", "instructions"])
def test_fingerprint_still_binds_effective_execution_inputs(tmp_path: Path, change: str) -> None:
    """收敛字段后保留模型参数、workspace 和实际指令的变化检测。"""
    first = build_runtime(tmp_path)
    second = build_runtime(tmp_path)
    if change == "temperature":
        second.environment.agent_config = second.environment.agent_config.model_copy(
            update={"model": ModelConfig(provider="openai", name="fake-model", temperature=0.7)}
        )
    elif change == "workspace":
        second.environment.workspace_root = tmp_path / "different"
    else:
        second.environment.context_input.system.slots[0].content = "different instructions"
    assert compute_environment_fingerprint(first) != compute_environment_fingerprint(second)


def test_fingerprint_binds_discovered_skill_body(tmp_path: Path) -> None:
    """目录启用时，正文变化影响新 runtime 的恢复版本。"""
    skill = tmp_path / ".agents/skills/review/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: review\ndescription: Review code\n---\nOriginal body", encoding="utf-8"
    )
    config = AgentConfig(
        name="skills",
        model="openai/fake-model",
        system="instructions",
        permissions={"workspace": str(tmp_path)},
        skills={"enabled": True},
    )
    original = RuntimeFactory.from_config(config, provider=StaticProvider())
    original_fingerprint = compute_environment_fingerprint(original)
    skill.write_text(
        "---\nname: review\ndescription: Review code\n---\nChanged body", encoding="utf-8"
    )
    changed = RuntimeFactory.from_config(config, provider=StaticProvider())
    assert compute_environment_fingerprint(original) == original_fingerprint
    assert compute_environment_fingerprint(changed) != original_fingerprint


@pytest.mark.parametrize(
    "override",
    [
        ProviderConfig(base_url="https://model.example/v2"),
        ProviderConfig(litellm_provider="anthropic"),
        ProviderConfig(headers={"X-Model-Revision": "new"}),
    ],
)
def test_fingerprint_binds_resolved_global_provider_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: ProviderConfig,
) -> None:
    """全局 provider 路由在 factory 合并后参与指纹，旧实例仍使用原投影。"""
    config = AgentConfig(
        name="global-provider",
        model="openai/fake-model",
        system="instructions",
        permissions={"workspace": str(tmp_path)},
    )
    monkeypatch.setattr(config_module, "_config", Config())
    original = RuntimeFactory.from_config(config, api_key="test-key")
    original_version = compute_environment_fingerprint(original)
    monkeypatch.setattr(config_module, "_config", Config(providers={"openai": override}))
    changed = RuntimeFactory.from_config(config, api_key="other-key")
    assert compute_environment_fingerprint(original) == original_version
    assert compute_environment_fingerprint(changed) != original_version


def test_fingerprint_uses_resolved_provider_without_api_key_or_shadowed_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式 endpoint 覆盖全局配置后，只比较实际路由且不绑定凭证。"""
    config = AgentConfig(
        name="global-provider",
        model=ModelConfig(
            provider="openai",
            name="fake-model",
            base_url="https://model.example/explicit",
        ),
        system="instructions",
        permissions={"workspace": str(tmp_path)},
    )
    monkeypatch.setattr(config_module, "_config", Config())
    original = RuntimeFactory.from_config(config, api_key="test-key")
    monkeypatch.setattr(
        config_module,
        "_config",
        Config(
            providers={"openai": ProviderConfig(base_url="https://model.example/shadowed")},
        ),
    )
    changed = RuntimeFactory.from_config(config, api_key="other-key")
    assert compute_environment_fingerprint(original) == compute_environment_fingerprint(changed)


def test_injected_provider_version_is_explicit_and_ignores_unused_route(tmp_path: Path) -> None:
    """host provider 不探测内部能力，只由显式版本和实际请求参数约束恢复。"""
    original = build_runtime(tmp_path)
    changed = build_runtime(tmp_path)
    changed.environment.agent_config = changed.environment.agent_config.model_copy(
        update={
            "model": ModelConfig(
                provider="unused",
                name="fake-model",
                base_url="https://unused.example",
            )
        }
    )
    assert compute_environment_fingerprint(original) == compute_environment_fingerprint(changed)
    changed.environment.provider_fingerprint = {"version": "new-model-route"}
    assert compute_environment_fingerprint(original) != compute_environment_fingerprint(changed)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "command",
        "args",
        "cwd",
        "env",
        "protocol",
        "output_schema",
        "tool_added",
        "timeout",
        "required",
        "trust",
        "filter",
    ],
)
async def test_mcp_fingerprint_binds_effective_catalog_and_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    from mcp import types

    from iris.harness import AgentRunner

    from ..mcp.fixtures.runtime import MCPPeer, mcp_agent

    peer = MCPPeer(monkeypatch)
    config = mcp_agent(tmp_path)
    first = AgentRunner.from_config(config, provider=StaticProvider())
    await first.aprepare()
    before = first.environment_fingerprint
    server = {"command": "fixture"}
    if change == "command":
        server["command"] = "different"
    elif change == "args":
        server["args"] = ["a", "b"]
    elif change == "cwd":
        server["cwd"] = str(tmp_path / "subdir")
    elif change == "env":
        server["env"] = {"IRIS_TEST_VALUE": "changed"}
    elif change == "protocol":
        peer.protocol_version = "2025-11-25"
    elif change == "output_schema":
        peer.tools = (peer.tools[0].model_copy(update={"output_schema": {"type": "object"}}),)
    elif change == "tool_added":
        peer.tools += (types.Tool(name="other", input_schema={"type": "object"}),)
    elif change == "timeout":
        server["tool_timeout_sec"] = 31
    elif change == "required":
        server["required"] = False
    elif change == "trust":
        config = config.model_copy(update={"mcp": config.mcp.model_copy(update={"overrides": {}})})
    else:
        server["disabled_tools"] = ["not-published"]
    config.mcp.path.write_text(json.dumps({"servers": {"test": server}}))
    second = AgentRunner.from_config(config, provider=StaticProvider())
    try:
        await second.aprepare()
        assert second.environment_fingerprint != before
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("trust", [False, True])
async def test_mcp_read_only_hint_only_binds_effective_local_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trust: bool,
) -> None:
    from mcp import types

    from iris.harness import AgentRunner

    from ..mcp.fixtures.runtime import MCPPeer, mcp_agent

    peer = MCPPeer(monkeypatch)
    config = mcp_agent(tmp_path, trust=trust)
    first = AgentRunner.from_config(config, provider=StaticProvider())
    await first.aprepare()
    peer.tools = (
        peer.tools[0].model_copy(
            update={"annotations": types.ToolAnnotations(read_only_hint=False)}
        ),
    )
    second = AgentRunner.from_config(config, provider=StaticProvider())
    await second.aprepare()
    assert (first.environment_fingerprint != second.environment_fingerprint) is trust
    await first.aclose()
    await second.aclose()


@pytest.mark.asyncio
async def test_mcp_portable_sources_and_diagnostics_do_not_change_effective_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from iris.agents import AgentMCPConfig
    from iris.harness import AgentRunner
    from iris.lifecycle import AgentRunRequest
    from iris.store import SQLiteStore

    from ..mcp.fixtures.runtime import MCPPeer, mcp_agent

    MCPPeer(monkeypatch)
    config = mcp_agent(tmp_path)
    secret = "synthetic-mcp-secret-512"
    monkeypatch.setenv("IRIS_FINGERPRINT_TOKEN", secret)
    config.mcp.path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "test": {
                        "type": "http",
                        "url": "https://example.invalid/mcp",
                        "headers": {"X-Test": "${IRIS_FINGERPRINT_TOKEN}"},
                        "disabled_tools": ["x", "y", "x"],
                    }
                }
            }
        )
    )
    first = AgentRunner.from_config(config, provider=StaticProvider())
    await first.aprepare()
    toml = tmp_path / "equivalent.toml"
    toml.write_text(
        '[mcp_servers.test]\ntype="streamable-http"\nurl="https://example.invalid/mcp"\n'
        'disabled_tools=["y","x"]\n[mcp_servers.test.http_headers]\nx-test="${env:IRIS_FINGERPRINT_TOKEN}"\n'
        "[mcp_servers.optional]\nrequired=false\nunsupported=true\n"
    )
    equivalent = config.model_copy(
        update={"mcp": AgentMCPConfig(path=toml, overrides=config.mcp.overrides)}
    )
    store_path = tmp_path / "fingerprint.db"
    store = SQLiteStore(store_path)
    second = AgentRunner.from_config(
        equivalent, store=store, provider=StaticProvider(text_response())
    )
    try:
        await second.start(AgentRunRequest(input="done", run_id="secret"))
        assert second.environment_fingerprint == first.environment_fingerprint
        assert secret not in caplog.text
        assert secret.encode() not in store_path.read_bytes()
        assert secret not in store.load_checkpoint("secret").model_dump_json()
        monkeypatch.setenv("IRIS_FINGERPRINT_TOKEN", "different-token")
        changed = AgentRunner.from_config(equivalent, provider=StaticProvider())
        await changed.aprepare()
        assert changed.environment_fingerprint != second.environment_fingerprint
        await changed.aclose()
    finally:
        await first.aclose()
        await second.aclose()
