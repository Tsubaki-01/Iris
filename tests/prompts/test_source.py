"""项目模板补齐、固定来源与真实进程竞争。"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path
from threading import Event

import pytest
from jinja2 import FileSystemLoader

import iris.prompts.source as prompt_module
from iris.exceptions import IrisTemplateError
from iris.prompts import PROMPT_IDS, PromptConfig, PromptSource


def test_initialization_preserves_existing_and_fills_missing(tmp_path: Path) -> None:
    root = tmp_path / ".iris" / "prompts"
    root.mkdir(parents=True)
    custom = root / "compaction.j2"
    custom.write_text("用户模板 {{ value }}", encoding="utf-8")
    assert PromptConfig().root == ".iris/prompts"
    source = PromptSource.initialize(tmp_path)
    assert source.root == root.resolve()
    assert len(PROMPT_IDS) == 13
    assert {path.stem for path in root.glob("*.j2")} == set(PROMPT_IDS)
    assert custom.read_text(encoding="utf-8") == "用户模板 {{ value }}"
    for prompt_id in PROMPT_IDS:
        if prompt_id != "compaction":
            assert (root / f"{prompt_id}.j2").read_bytes() == files("iris.prompts").joinpath(
                f"{prompt_id}.j2"
            ).read_bytes()


def test_snapshot_keeps_sources_but_uses_current_variables(tmp_path: Path) -> None:
    source = PromptSource.initialize(tmp_path, root="chosen-prompts")
    entry = source.root / "compaction.j2"
    entry.write_text("{% include selected %} {{ value }}", encoding="utf-8")
    dependency = source.root / "selected.txt"
    dependency.write_text("old", encoding="utf-8")
    snapshot = source.snapshot()
    dependency.write_text("new", encoding="utf-8")
    entry.write_text("changed {{ value }}", encoding="utf-8")
    assert snapshot.render("compaction", {"selected": "selected.txt", "value": 1}) == "old 1"
    assert snapshot.render("compaction", {"selected": "selected.txt", "value": 2}) == "old 2"
    assert source.snapshot().render("compaction", {"value": 3}) == "changed 3"


def test_deleted_project_template_never_falls_back_to_seed(tmp_path: Path) -> None:
    source = PromptSource.initialize(tmp_path)
    (source.root / "compaction.j2").unlink()
    with pytest.raises(IrisTemplateError):
        source.snapshot().render("compaction", {})
    PromptSource.initialize(tmp_path)
    assert (source.root / "compaction.j2").is_file()


def test_initialization_reports_target_path(tmp_path: Path) -> None:
    root = tmp_path / "blocked"
    root.write_text("file", encoding="utf-8")
    with pytest.raises(IrisTemplateError) as error:
        PromptSource.initialize(tmp_path, root="blocked")
    assert error.value.context["path"] == str(root)


def test_decision_seeds_preserve_existing_instructions(tmp_path: Path) -> None:
    snapshot = PromptSource.initialize(tmp_path).snapshot()
    assert snapshot.render("memory_recall_instruction", {"candidate_index": 4}) == (
        "评价 state.memories[4] 对回答 state.query 的帮助程度。"
    )
    assert snapshot.render("tool_discovery_instruction", {"query_index": 2}) == (
        "Select the best tool from state.tools for state.queries[2]."
    )


@pytest.mark.parametrize("winner", ["initializer", "user"])
def test_two_process_initialization_never_replaces_published_file(
    tmp_path: Path, winner: str
) -> None:
    """两个进程已决定补种时，后发布者仍不能替换先完成者或用户内容。"""
    coordination = tmp_path / "barriers"
    coordination.mkdir()
    workspace = tmp_path / "workspace"
    script = Path(__file__).with_name("_initialize_worker.py")
    processes = [
        subprocess.Popen(
            [sys.executable, str(script), str(workspace), str(coordination), str(label)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for label in (1, 2)
    ]
    try:
        deadline = time.monotonic() + 15
        while not all((coordination / f"{label}.ready").exists() for label in (1, 2)):
            assert all(process.poll() is None for process in processes)
            assert time.monotonic() < deadline, "初始化进程未到达发布屏障"
            time.sleep(0.01)
        root = workspace / ".iris" / "prompts"
        target = root / "compaction.j2"
        seed = files("iris.prompts").joinpath("compaction.j2").read_bytes()
        expected = b"1\n" + seed
        if winner == "user":
            expected = "用户在初始化期间写入的完整正文".encode()
            target.write_bytes(expected)
        (coordination / "1.release").touch()
        stdout, stderr = processes[0].communicate(timeout=15)
        assert processes[0].returncode == 0, stdout + stderr
        assert target.read_bytes() == expected
        (coordination / "2.release").touch()
        stdout, stderr = processes[1].communicate(timeout=15)
        assert processes[1].returncode == 0, stdout + stderr
        assert target.read_bytes() == expected
        assert {path.name for path in root.iterdir()} == {
            f"{prompt_id}.j2" for prompt_id in PROMPT_IDS
        }
        for prompt_id in PROMPT_IDS:
            if prompt_id != "compaction":
                assert (root / f"{prompt_id}.j2").read_bytes() == files("iris.prompts").joinpath(
                    f"{prompt_id}.j2"
                ).read_bytes()
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate()


def test_snapshot_during_other_initializer_cleanup_has_only_template_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """一方初始化后立即快照，另一方清理暂存文件不会破坏已枚举的来源。"""
    staged = Event()
    release = Event()
    original_link = prompt_module.os.link
    original_list_templates = FileSystemLoader.list_templates

    def link(source: Path, target: Path) -> None:
        if target.name == "compaction.j2" and not staged.is_set():
            staged.set()
            assert release.wait(timeout=10), "等待另一初始化方快照超时"
        original_link(source, target)

    monkeypatch.setattr(prompt_module.os, "link", link)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(PromptSource.initialize, tmp_path)
        try:
            assert staged.wait(timeout=10), "初始化未到达发布边界"
            source = PromptSource.initialize(tmp_path)

            def list_templates(loader: FileSystemLoader) -> list[str]:
                names = original_list_templates(loader)
                release.set()
                assert pending.result(timeout=10).root == source.root
                return names

            monkeypatch.setattr(FileSystemLoader, "list_templates", list_templates)
            snapshot = source.snapshot()
            assert snapshot.render("tool_discovery_instruction", {"query_index": 0}) == (
                "Select the best tool from state.tools for state.queries[0]."
            )
        finally:
            release.set()
