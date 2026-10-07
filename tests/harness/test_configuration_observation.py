"""配置描述必须来自实际装配快照，不在观察时回读文件。"""

from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.harness import AgentRunner, AgentRunRequest
from iris.observability.facts import ConfigurationApplied, SourceAdopted, bind_fact_scope
from iris.skill.tool import LoadSkillInput, LoadSkillTool
from iris.store import SQLiteStore
from iris.tools import ToolExecutionContext

from ..skill.test_skill_tool import _registry
from .fakes import RecordingPublisher, StaticProvider, text_response
from .test_runner_subagent import StreamingStaticProvider


@pytest.mark.asyncio
async def test_configuration_description_preserves_loaded_sources_and_actual_activation(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("name: test\nmodel: fake/model\nsystem: original\n", encoding="utf-8")
    publisher = RecordingPublisher()
    runner = AgentRunner.from_config_path(
        config_path,
        provider=StreamingStaticProvider(text_response("done")),
        live_publisher=publisher,
    )
    before = runner.describe_configuration()
    assert before.agent_config.system == "original"
    assert any(
        document.text == config_path.read_text(encoding="utf-8")
        for document in before.source_documents
    )
    config_path.write_text("name: test\nmodel: fake/model\nsystem: changed\n", encoding="utf-8")
    prompt = runner.runtime.environment.prompt_source.root / "compaction.j2"
    old_prompt = prompt.read_bytes().decode("utf-8")
    prompt.write_text("new compaction", encoding="utf-8")
    after = runner.describe_configuration()
    assert after == before
    assert any(document.text == old_prompt for document in after.source_documents)
    await runner.start(AgentRunRequest(input="hello", run_id="run"))
    applied = [fact for fact in publisher.facts if isinstance(fact, ConfigurationApplied)]
    assert len(applied) == 1
    assert applied[0].configuration_snapshot_id == before.configuration_snapshot_id
    assert applied[0].configuration == before
    assert applied[0].activation_id == next(
        event.activation_id
        for event in runner.list_events("run")
        if event.kind == "activation.started"
    )
    replacement = AgentRunner.from_config_path(
        config_path, provider=StaticProvider(text_response("done"))
    )
    assert (
        replacement.describe_configuration().configuration_snapshot_id
        != before.configuration_snapshot_id
    )
    assert replacement.describe_configuration().agent_config.system == "changed"
    await runner.aclose()
    await replacement.aclose()


def test_python_configuration_does_not_invent_raw_yaml(tmp_path: Path) -> None:
    runner = AgentRunner.from_config(
        AgentConfig(name="python", model="fake/model", system="rules"),
        config_path=tmp_path / "not-loaded.yaml",
        provider=StaticProvider(text_response("done")),
    )
    view = runner.describe_configuration()
    assert view.source_completeness == "effective_only"
    assert not any(document.kind == "agent" for document in view.source_documents)


@pytest.mark.asyncio
async def test_loaded_skill_adoption_records_current_returned_text(tmp_path: Path) -> None:
    """Catalog 发现之后正文改变，采用事实指向本次实际读取结果。"""
    registry, path = _registry(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "\nnew instruction", encoding="utf-8")
    facts = []
    with bind_fact_scope(facts.append, run_id="r", session_id="s", activation_id="a"):
        result = await LoadSkillTool(registry).arun(
            LoadSkillInput(name="example-skill"), ToolExecutionContext(workspace_root=tmp_path)
        )
    [adopted] = facts
    assert isinstance(adopted, SourceAdopted)
    assert adopted.documents[0].text == result.model_content
    assert adopted.documents[0].text.endswith("new instruction")
    assert adopted.run_id == "r" and adopted.source_kind == "skill"


def test_description_reports_injected_store_instead_of_declared_path(tmp_path: Path) -> None:
    """实际 store binding 是注入对象，配置声明不会被伪装为采用结果。"""
    config = AgentConfig.model_validate(
        {
            "name": "binding",
            "model": "fake/model",
            "system": "rules",
            "session": {"backend": "sqlite", "path": "unused.db"},
        }
    )
    store = SQLiteStore(tmp_path / "actual.db")
    runner = AgentRunner.from_config(
        config,
        config_path=tmp_path / "agent.yaml",
        store=store,
        provider=StaticProvider(text_response("done")),
    )
    description = runner.describe_configuration()
    assert description.storage.path == str(store.path.resolve())
    assert description.agent_config.session.path == "unused.db"
    assert (
        next(item for item in description.dependencies if item.kind == "store").origin == "injected"
    )
    assert not (tmp_path / "unused.db").exists()
