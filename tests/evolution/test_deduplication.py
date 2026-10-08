"""档案派生字段和经验选材的无损去重。"""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from iris.evolution.history import PublicationRecord
from iris.evolution.materials import EvolutionMaterialStore
from iris.evolution.models import EvolutionMaterial, EvolutionResult, RevisionEvidence
from iris.exceptions import IrisEvolutionError
from iris.message import LLMRequest

from .test_materials import _block, _issue, _source
from .test_project_skill import append_source, prepare
from .test_publications import reopen
from .test_strategy_revision import issue_output


@pytest.mark.parametrize("stage", ["experience", "revision"])
def test_publication_projects_evidence_and_request_without_storing_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """完整读取可重组原 DTO，但两个 JSON 列都没有派生副本。"""
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    block = _block(source, 2, 5, terminal=5)
    materials = (
        EvolutionMaterial(
            source=source,
            start_message_count=2,
            end_message_count=5,
            records=(
                block.records[0],
                block.records[1].model_copy(update={"text": "  \n"}),
                block.records[2],
            ),
        ),
    )
    issue = _issue(source)
    if stage == "revision":
        store.enqueue_revision(issue)
        record = PublicationRecord(
            stage="revision",
            revision_id=issue.id,
            origin="experience",
            description=issue.description,
            targets=issue.targets,
            request=issue,
            evidence_refs=issue.evidence,
        )
    else:
        record = PublicationRecord(
            stage="experience",
            origin="experience",
            description="整理本批材料",
            materials=materials,
            proposed_issue=issue,
            evidence_refs=tuple(
                RevisionEvidence(ref=entry.ref, quote=entry.text)
                for entry in materials[0].records
                if entry.text.strip()
            ),
        )
    store.save_publication(record)
    with sqlite3.connect(store.path) as database:
        state_json, detail_json = database.execute(
            "SELECT state_json,detail_json FROM publications "
            "JOIN publication_details ON publication_id=id"
        ).fetchone()
    for payload in (json.loads(state_json), json.loads(detail_json)):
        assert "evidence_refs" not in payload
        assert "request" not in payload
    if stage == "experience":
        assert json.loads(detail_json)["proposed_issue"]["id"] == issue.id
    else:
        store.settle_revision(
            issue.id, EvolutionResult(stage="revision", status="no_change", revision_id=issue.id)
        )

    restarted = EvolutionMaterialStore(tmp_path)
    connect = sqlite3.connect
    connections: list[sqlite3.Connection] = []

    def tracked_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    assert restarted.get_publication(record.publication_id) == record
    assert len(connections) == 1
    connections.clear()
    assert restarted.list_unsettled_publications() == (record,)
    assert len(connections) == 1


@pytest.mark.parametrize("capacity", [0, 1, 2, 3])
def test_prepare_serializes_each_material_once_and_preserves_prefix_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capacity: int
) -> None:
    """用实际请求文本长度锁定边界，同时统计真实材料序列化。"""
    service, provider, _ = prepare(tmp_path)
    append_source(service, "second")
    append_source(service, "third")
    materials = service.store.read_pending(
        allowed_sources=frozenset({("host", "run"), ("host", "second"), ("host", "third")})
    ).items
    payloads = [
        json.dumps(
            {
                "current_skill": "",
                "skill_max_chars": service.config.skill_max_chars,
                "materials": [item.model_dump(mode="json") for item in materials[:count]],
                "open_targets": {
                    "prompt": service.config.prompt_targets,
                    "config": service.config.config_targets,
                },
            },
            ensure_ascii=False,
        )
        for count in range(1, 4)
    ]
    budget = len(payloads[capacity - 1]) if capacity else len(payloads[0]) - 1
    service.config = service.config.model_copy(update={"input_budget_tokens": budget})
    estimated: list[str] = []

    def estimate(request: LLMRequest) -> int:
        estimated.append(request.messages[1].text)
        return len(request.messages[1].text)

    provider.estimate_input_tokens = estimate
    serialized: list[str] = []
    dump = EvolutionMaterial.model_dump

    def tracked_dump(material: EvolutionMaterial, **kwargs: Any) -> dict[str, Any]:
        serialized.append(material.source.run_id)
        return dump(material, **kwargs)

    monkeypatch.setattr(EvolutionMaterial, "model_dump", tracked_dump)
    if capacity == 0:
        with pytest.raises(IrisEvolutionError, match="预算"):
            service._prepare(materials)
    else:
        baseline, selected, request = service._prepare(materials)
        assert baseline is None
        assert selected == materials[:capacity]
        assert request.messages[1].text == payloads[capacity - 1]
    attempted = min(capacity + 1, len(materials))
    assert estimated == payloads[:attempted]
    assert serialized == [item.source.run_id for item in materials[:attempted]]
    assert provider.requests == []


@pytest.mark.asyncio
async def test_deduplicated_experience_recovers_proposed_issue_after_confirmation(
    tmp_path: Path,
) -> None:
    """确认已写入而消费失败后，提案独立保留且只创建一个请求。"""
    service, provider, scope = prepare(tmp_path, prompt_targets=["compaction"])
    provider.output = issue_output()
    provider.output["body"] = "# 项目经验\n\n使用 uv 管理依赖。"
    with sqlite3.connect(service.store.path) as database:
        database.execute(
            "CREATE TRIGGER fail_settle BEFORE INSERT ON progress "
            "BEGIN SELECT RAISE(ABORT, 'settlement interrupted'); END"
        )
    with pytest.raises(IrisEvolutionError, match="settlement interrupted"):
        await service.maintain_cycle(scope=scope)
    publication_id = service.list_publications().items[0].publication_id
    record = service.get_publication(publication_id)
    assert record is not None and record.proposed_issue is not None
    assert record.publication_state == "confirmed" and not record.settled
    issue = record.proposed_issue
    assert service.get_revision_request(issue.id) is None
    with sqlite3.connect(service.store.path) as database:
        database.execute("DROP TRIGGER fail_settle")
    restarted = reopen(service)
    result = await restarted.maintain_cycle(scope=scope)
    assert result.publication_id == publication_id and result.status == "updated"
    assert restarted.get_revision_request(issue.id) == issue
    assert len(restarted.list_revision_requests().items) == 1
    assert len(provider.requests) == 1
