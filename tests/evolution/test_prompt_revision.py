"""Prompt B 候选在原冻结源中试渲染，通过后才发布。"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from jinja2 import FileSystemLoader

from iris.evolution.config import EvolutionConfig
from iris.evolution.models import RevisionTarget
from iris.evolution.revision import (
    PromptTarget,
    check_targets,
    prepare_candidate,
    prepare_revision,
    publish_revision,
)
from iris.exceptions import IrisEvolutionError
from iris.message import LLMResponse, TextBlock
from iris.prompts import PromptSource


def response(payload: dict[str, object]) -> LLMResponse:
    return LLMResponse(
        provider="test", finish_reason="stop", content=[TextBlock(text=json.dumps(payload))]
    )


def test_prompt_candidate_uses_original_dependencies_and_keeps_old_snapshot(tmp_path: Path) -> None:
    source = PromptSource.initialize(tmp_path)
    target = source.root / "compaction_input.j2"
    target.write_text("旧 {{ serialized_history }}", encoding="utf-8")
    dependency = source.root / "part.j2"
    dependency.write_text("原依赖", encoding="utf-8")
    snapshot = source.snapshot()
    context = prepare_revision(
        prompt_snapshot=snapshot,
        targets=(RevisionTarget(kind="prompt", name="compaction_input"),),
        prompt_targets=(
            PromptTarget(
                "compaction_input", "压缩输入；保留完整历史变量", {"serialized_history": "样例"}
            ),
        ),
        config_target=None,
    )
    dependency.unlink()
    body = '{% include "part.j2" %} 新 {{ serialized_history }}'
    candidate = prepare_candidate(
        response(
            {"action": "prompt", "target": "compaction_input", "body": body, "reason": "明确指令"}
        ),
        context,
    )
    assert target.read_text(encoding="utf-8") == "旧 {{ serialized_history }}"
    assert snapshot.render("compaction_input", {"serialized_history": "当前"}) == "旧 当前"
    assert publish_revision(candidate)
    assert target.read_text(encoding="utf-8") == body


@pytest.mark.parametrize("body", ["{{ missing }}", '{% include "missing.j2" %}', "{% invalid %}"])
def test_invalid_prompt_candidate_leaves_target_unchanged(tmp_path: Path, body: str) -> None:
    source = PromptSource.initialize(tmp_path)
    path = source.root / "compaction.j2"
    baseline = path.read_text(encoding="utf-8")
    context = prepare_revision(
        prompt_snapshot=source.snapshot(),
        targets=(RevisionTarget(kind="prompt", name="compaction"),),
        prompt_targets=(PromptTarget("compaction", "普通摘要，不要求 JSON", {}),),
        config_target=None,
    )
    with pytest.raises(IrisEvolutionError):
        prepare_candidate(
            response({"action": "prompt", "target": "compaction", "body": body, "reason": "修改"}),
            context,
        )
    assert path.read_text(encoding="utf-8") == baseline


def test_prompt_publish_preserves_concurrent_manual_change(tmp_path: Path) -> None:
    source = PromptSource.initialize(tmp_path)
    context = prepare_revision(
        prompt_snapshot=source.snapshot(),
        targets=(RevisionTarget(kind="prompt", name="compaction"),),
        prompt_targets=(PromptTarget("compaction", "摘要", {}),),
        config_target=None,
    )
    candidate = prepare_candidate(
        response({"action": "prompt", "target": "compaction", "body": "新策略", "reason": "调整"}),
        context,
    )
    path = source.root / "compaction.j2"
    path.write_text("用户更新", encoding="utf-8")
    assert not publish_revision(candidate)
    assert path.read_text(encoding="utf-8") == "用户更新"


def test_prompt_publication_does_not_expose_temporary_sources_to_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = PromptSource.initialize(tmp_path)
    path = source.root / "compaction.j2"
    context = prepare_revision(
        prompt_snapshot=source.snapshot(),
        targets=(RevisionTarget(kind="prompt", name="compaction"),),
        prompt_targets=(PromptTarget("compaction", "摘要", {}),),
        config_target=None,
    )
    candidate = prepare_candidate(
        response({"action": "prompt", "target": "compaction", "body": "新策略", "reason": "调整"}),
        context,
    )
    staged, release = Event(), Event()
    replace = Path.replace
    list_templates = FileSystemLoader.list_templates

    def paused_replace(temporary: Path, target: Path) -> Path:
        if target == path:
            staged.set()
            assert release.wait(5)
        return replace(temporary, target)

    monkeypatch.setattr(Path, "replace", paused_replace)
    with ThreadPoolExecutor(max_workers=1) as executor:
        publication = executor.submit(publish_revision, candidate)

        def enumerated_before_publication(loader: FileSystemLoader) -> list[str]:
            names = list_templates(loader)
            release.set()
            assert publication.result(timeout=5)
            return names

        try:
            assert staged.wait(5)
            monkeypatch.setattr(FileSystemLoader, "list_templates", enumerated_before_publication)
            snapshot = source.snapshot()
            assert snapshot.render("compaction", {}) == "新策略"
        finally:
            release.set()


def test_target_scope_and_no_change_do_not_allow_other_named_prompts(tmp_path: Path) -> None:
    config = EvolutionConfig(prompt_targets=["compaction"])
    target = RevisionTarget(kind="prompt", name="compaction")
    check_targets((target,), config)
    with pytest.raises(IrisEvolutionError):
        check_targets((RevisionTarget(kind="prompt", name="goal_context"),), config)
    source = PromptSource.initialize(tmp_path)
    context = prepare_revision(
        prompt_snapshot=source.snapshot(),
        targets=(target,),
        prompt_targets=(PromptTarget("compaction", "摘要", {}),),
        config_target=None,
    )
    with pytest.raises(IrisEvolutionError):
        prepare_candidate(
            response(
                {"action": "prompt", "target": "memory_flush", "body": "文本", "reason": "修改"}
            ),
            context,
        )
    candidate = prepare_candidate(response({"action": "no_change", "reason": "无需修改"}), context)
    assert candidate.action == "no_change" and publish_revision(candidate)
