"""通过真实运行事实验证执行树和可搬运产物导出。"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from iris.agents import AgentConfig
from iris.harness import AgentRunner, SessionHistory, collect_trajectory, write_trajectory
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import AgentRunRequest, RunStopReason
from iris.lifecycle.history import ChildRunPage, RunCursor
from iris.message import ImageBlock, ImageFileRef, TextBlock, ToolUseBlock
from iris.providers import CompletionProvider
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolArtifact, ToolRegistry, ToolResult

from .fakes import (
    BlockingProvider,
    StaticProvider,
    build_runtime,
    text_response,
    tool_batch_response,
    tool_response,
)


@pytest.mark.asyncio
async def test_archive_keeps_effective_context_settings(tmp_path: Path) -> None:
    """保存本次采用的上下文策略与压缩预算，离线比较无需回读 profile。"""
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: context-test\nmodel: openai/fake\nsystem: 执行任务。\n"
        "context_policy:\n  old_result_preview_chars: 128\n"
        "compaction:\n  input_budget_tokens: 4096\n",
        encoding="utf-8",
    )
    runner = AgentRunner.from_config_path(path, provider=StaticProvider(text_response()))
    await runner.start(AgentRunRequest(input="执行", session_id="sample"))
    await runner.aclose()
    path.unlink()
    trace = write_trajectory(collect_trajectory(runner, "sample"), tmp_path / "archive")
    assert trace["configuration"]["context_policy"]["old_result_preview_chars"] == 128
    assert trace["configuration"]["compaction"]["input_budget_tokens"] == 4096


@pytest.mark.asyncio
@pytest.mark.parametrize("sqlite", [False, True])
async def test_empty_and_paginated_multiturn_after_close(tmp_path: Path, sqlite: bool) -> None:
    store = SQLiteStore(tmp_path / "lifecycle.db") if sqlite else InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path, provider=StaticProvider(*(text_response(str(i)) for i in range(51)))
        ),
        store=store,
    )
    empty = collect_trajectory(runner, "sample")
    assert empty["root_run_ids"] == []
    assert empty["sessions"]["sample"]["messages"] == []
    for index in range(51):
        await runner.start(
            AgentRunRequest(input=f"input {index}", session_id="sample", run_id=f"r{index}")
        )
    await runner.aclose()
    document = collect_trajectory(runner, "sample")
    assert document["root_run_ids"] == [f"r{i}" for i in range(51)]
    assert len(document["runs"]) == 51
    assert list(document["sessions"]) == ["sample"]
    assert len(document["sessions"]["sample"]["messages"]) == 102
    assert document["export"]["status"] == "complete"
    assert document["runs"]["r0"]["result"]["run"]["usage"]["total_tokens"] == 5
    for record in document["runs"].values():
        sequences = [event["sequence"] for event in record["events"]]
        assert sequences == sorted(sequences)


@pytest.mark.asyncio
async def test_children_retain_parent_tool_relationships(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, child in [("root", "child"), ("child", None)]:
        config = f"name: {name}\nmodel: openai/fake\nsystem: 执行任务。\n"
        if child:
            config += f"tools:\n  subagent: {name}-routes.yaml\n"
            (tmp_path / f"{name}-routes.yaml").write_text(
                f"default: {child}\nagents:\n  {child}:\n"
                f"    path: {child}.yaml\n    description: 执行子任务\n",
                encoding="utf-8",
            )
        (tmp_path / f"{name}.yaml").write_text(config, encoding="utf-8")

    def child_provider(config: AgentConfig, *, config_path: Path) -> CompletionProvider:
        return StaticProvider(text_response())

    runner = AgentRunner.from_config_path(
        tmp_path / "root.yaml",
        child_provider_factory=child_provider,
        provider=StaticProvider(
            tool_batch_response(
                *(
                    ToolUseBlock(id=f"delegate-{i}", name="subagent", input={"prompt": "完成"})
                    for i in range(2)
                )
            ),
            text_response(),
        ),
    )
    result = await runner.start(AgentRunRequest(input="执行", run_id="root", session_id="sample"))
    await runner.aclose()
    assert result.run.stop_reason is RunStopReason.COMPLETED
    original_list = runner.list_child_runs

    def one_child(parent_run_id: str, *, after: RunCursor | None = None) -> ChildRunPage:
        return original_list(parent_run_id, after=after, limit=1)

    monkeypatch.setattr(runner, "list_child_runs", one_child)
    document = collect_trajectory(runner, "sample")
    assert len(document["runs"]) == len(document["sessions"]) == 3
    parents = [r["parent"] for r in document["runs"].values() if r["parent"] is not None]
    assert {parent["parent_tool_call_id"] for parent in parents} == {"delegate-0", "delegate-1"}
    assert all(parent["parent_run_id"] == "root" for parent in parents)
    assert document["root_run_ids"] == ["root"]


@pytest.mark.asyncio
async def test_active_without_result_then_cancelled(tmp_path: Path) -> None:
    provider = BlockingProvider()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=provider), store=InMemoryLifecycleStore()
    )
    task = asyncio.create_task(
        runner.start(AgentRunRequest(input="等待", session_id="sample", run_id="active"))
    )
    await provider.started.wait()
    active = collect_trajectory(runner, "sample")
    assert active["runs"]["active"]["result"] is None
    assert active["export"]["status"] == "partial"
    await runner.cancel("active")
    await task
    await runner.aclose()
    cancelled = collect_trajectory(runner, "sample")
    assert cancelled["runs"]["active"]["run"]["stop_reason"] == "cancelled"
    assert cancelled["export"]["status"] == "complete"


@pytest.mark.asyncio
async def test_waiting_resume_does_not_duplicate_run(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: ask\nmodel: openai/fake\nsystem: 询问。\ntools:\n  builtin: [human.ask]\n",
        encoding="utf-8",
    )
    runner = AgentRunner.from_config_path(
        path,
        provider=StaticProvider(
            tool_response(ToolUseBlock(id="ask", name="ask_question", input={"question": "选？"})),
            text_response(),
        ),
    )
    waiting = await runner.start(AgentRunRequest(input="开始", session_id="s", run_id="r"))
    before = collect_trajectory(runner, "s")
    assert before["runs"]["r"]["run"]["phase"] == "waiting"
    await runner.resume(
        "r",
        interaction_id=waiting.run.pending_interaction_id,
        response=QuestionInteractionResponse(answer="继续"),
    )
    await runner.aclose()
    after = collect_trajectory(runner, "s")
    assert after["root_run_ids"] == ["r"]
    assert after["runs"]["r"]["run"]["usage"]["total_tokens"] == 13


@pytest.mark.asyncio
async def test_portable_images_full_text_and_missing_asset(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = workspace / "image.png"
    Image.new("RGB", (2, 2), "red").save(original)
    full_text = workspace / "full.txt"
    full_text.write_text("完整工具输出", encoding="utf-8")
    (workspace / "unrelated.txt").write_text("不导出", encoding="utf-8")
    reference = ImageFileRef(path=original, mime_type="image/png", width=2, height=2)
    image = ImageBlock(original=reference, model=reference)
    registry = ToolRegistry()

    async def produce() -> ToolResult:
        """返回正文、图片和明确的全文文件。"""
        return ToolResult(
            tool_use_id="produce",
            tool_name="produce",
            content=[TextBlock(text="预览"), image],
            artifact=ToolArtifact(path=original, text_path=full_text),
            data={"output_truncated": True, "stdout": "已保留的头尾"},
        )

    registry.register_function(produce)
    runner = AgentRunner(
        store=InMemoryLifecycleStore(),
        runtime=build_runtime(
            workspace,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="produce", name="produce", input={})), text_response()
            ),
        ),
    )
    await runner.start(AgentRunRequest(input="生成", session_id="sample", run_id="run"))
    await runner.aclose()
    branch = SessionHistory(runner.store).fork("run")
    branch_trace = collect_trajectory(runner, branch.session_id)
    assert branch_trace["root_run_ids"] == []
    assert {item["source_path"] for item in branch_trace["assets"]} == {
        str(original),
        str(full_text),
    }
    document = collect_trajectory(runner, "sample")
    bundle = tmp_path / "archive"
    saved = write_trajectory(document, bundle)
    assert saved["export"]["status"] == "complete"
    assert len(saved["assets"]) == 2
    assert saved["runs"]["run"]["tool_calls"][0]["result"]["data"]["output_truncated"]
    moved = tmp_path / "moved"
    shutil.copytree(bundle, moved)
    shutil.rmtree(workspace)
    loaded = json.loads((moved / "trace.json").read_text(encoding="utf-8"))
    for asset in loaded["assets"]:
        copied = moved / asset["path"]
        assert copied.is_file()
        if copied.suffix == ".txt":
            assert copied.read_text(encoding="utf-8") == "完整工具输出"
        else:
            with Image.open(copied) as picture:
                assert picture.size == (2, 2)
    partial = write_trajectory(document, tmp_path / "missing")
    assert partial["export"]["status"] == "partial"
    assert all(asset["error"] for asset in partial["assets"])
    assert document["export"]["status"] == "complete"
    blocked = tmp_path / "file"
    blocked.write_text("占用路径", encoding="utf-8")
    failed = write_trajectory(document, blocked)
    assert failed["export"]["status"] == "failed"
