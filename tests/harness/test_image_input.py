"""已保存图片通过 SDK、入队与 durable 输入恢复的同一条路径。"""

from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
from threading import get_ident

import pytest
from PIL import Image
from pydantic import ValidationError

import iris.harness.runner as runner_module
from iris.context import ContextBuildScope, ContextSnapshot
from iris.exceptions import IrisImageError, IrisRunStateError
from iris.harness import AgentRunner, SessionEvent, SessionHistory, SessionManager, SubmissionEvent
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import AgentRunRequest, LifecycleStore, RunEvent, RunEventKind, RunStopReason
from iris.message import DataBlock, ImageBlock, LLMRequest, LLMResponse, TextBlock, ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import AskQuestionTool, ToolCapability, ToolRegistry
from iris.tools._paths import safe_path_segment
from iris.utils.images import SavedImage

from .fakes import (
    BlockingProvider,
    CountingAgentRuntime,
    StaticProvider,
    build_runtime,
    text_response,
    tool_response,
)
from .test_runner_subagent import ChildProviders, _parent_provider, _write_configs


def _png(color: str = "red") -> bytes:
    """生成不同导入快照可辨认的小图。"""
    output = BytesIO()
    with Image.new("RGB", (8, 5), color) as image:
        image.save(output, format="PNG")
    return output.getvalue()


@pytest.fixture(params=["memory", "sqlite"])
def image_store(request: pytest.FixtureRequest, tmp_path: Path) -> LifecycleStore:
    """同一输入序列化契约覆盖两种 store。"""
    return (
        SQLiteStore(tmp_path / "image-input.db")
        if request.param == "sqlite"
        else InMemoryLifecycleStore()
    )


@pytest.mark.asyncio
async def test_import_saves_workspace_path_and_bytes_without_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """导入离开 event loop，源文件改动不影响缓存，也不创建 session run。"""
    store = InMemoryLifecycleStore()
    provider = StaticProvider()
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    original = _png()
    (tmp_path / "source.png").write_bytes(original)
    main_thread = get_ident()
    save = runner_module.save_image

    def save_in_worker(source: Path | bytes, *, cache_dir: Path) -> SavedImage:
        """确认整个图片导入在 worker thread 处理。"""
        assert get_ident() != main_thread
        return save(source, cache_dir=cache_dir)

    monkeypatch.setattr(runner_module, "save_image", save_in_worker)
    first = await runner.import_image(Path("source.png"), session_id="图/一", name="source")
    second = await runner.import_image(original, session_id="图/二", name="bytes")
    (tmp_path / "source.png").write_bytes(_png("blue"))
    later = await runner.import_image(Path("source.png"), session_id="图/一")

    assert first.model.path.parent == tmp_path / ".iris/image-cache" / safe_path_segment("图/一")
    assert second.model.path.parent.name == safe_path_segment("图/二")
    assert first.original == first.model
    assert first.original.path.is_absolute() and first.original.path.read_bytes() == original
    assert second.model.path.read_bytes() == original
    assert later.model.path != first.model.path
    assert later.model.path.read_bytes() == _png("blue")
    assert first.name == "source" and second.name == "bytes"
    assert store.load_session_lane("图/一") is None
    assert store.load_session("图/一").messages == [] and provider.requests == []
    with pytest.raises(IrisImageError):
        await runner.import_image(b"not an image", session_id="图/一")
    with pytest.raises(IrisRunStateError, match="session_id"):
        await runner.import_image(original, session_id=" ")


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", " \n", [], [TextBlock(text="\t ")]])
async def test_empty_input_is_rejected_by_both_public_admission_paths(
    tmp_path: Path, content: str | list[DataBlock]
) -> None:
    """字符串与块列表遵守同一个至少有文字或图片的条件。"""
    with pytest.raises(ValidationError, match="input"):
        AgentRunRequest(input=content)
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path), store=InMemoryLifecycleStore()), "empty"
    )
    with pytest.raises(IrisRunStateError, match="input"):
        await manager.submit(content)
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("pure_image", [False, True])
async def test_start_keeps_order_and_projects_text_for_context_and_fork_point(
    tmp_path: Path, image_store: LifecycleStore, pure_image: bool
) -> None:
    """完整输入进入请求和历史，展示与动态 source 只接收文字。"""
    scopes: list[ContextBuildScope] = []

    class Source:
        """记录 source 接收到的纯文字视图。"""

        async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
            """保存作用域，不向模型追加额外资料。"""
            scopes.append(scope)
            return ContextSnapshot()

    provider = StaticProvider(text_response())
    runtime = CountingAgentRuntime(build_runtime(tmp_path, provider=provider))
    runtime.environment.context_source = Source()
    runner = AgentRunner(runtime=runtime, store=image_store)
    first = await runner.import_image(_png(), session_id="main", name="first.png")
    second = await runner.import_image(_png("blue"), session_id="main", name="second.png")
    content: list[DataBlock] = (
        [first, second]
        if pure_image
        else [TextBlock(text="先看"), first, TextBlock(text="再比较"), second]
    )
    result = await runner.start(AgentRunRequest(input=content, session_id="main", run_id="image"))

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert image_store.load_run("image").request.input == content
    assert image_store.load_session("main").messages[0].content == content
    assert provider.requests[0].messages[1].content == content
    assert runtime.activations[0].run_input == content
    assert scopes[0].run_input == ("" if pure_image else "先看\n再比较")
    point = SessionHistory(image_store).list_fork_points("main").items[0]
    assert point.input == (
        "[image: first.png]\n[image: second.png]" if pure_image else "先看\n再比较"
    )


@pytest.mark.asyncio
async def test_manager_idle_steer_and_follow_up_keep_imported_blocks(tmp_path: Path) -> None:
    """工具循环、图片 steer 与后续文字提问保图，delivery 仍晚于 durable commit。"""

    class ToolThenTextProvider(BlockingProvider):
        """首步阻塞后调用工具，后续步骤正常回答。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """沿用同步点，仅将首个响应换成普通工具调用。"""
            response = await super().complete(request)
            if len(self.requests) == 1:
                return tool_response(ToolUseBlock(id="note", name="get_note", input={}))
            return response

    def get_note() -> str:
        """提供图片比较所需的文字备注。"""
        return "compare the two colors"

    provider = ToolThenTextProvider(text_response())
    registry = ToolRegistry()
    registry.register_function(get_note, description="查看备注", capabilities={ToolCapability.READ})
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=provider), store=store
    )
    first = await runner.import_image(_png(), session_id="managed", name="first")
    second = await runner.import_image(_png("blue"), session_id="managed", name="second")
    initial: list[DataBlock] = [first]
    steering: list[DataBlock] = [TextBlock(text="比较"), second, first]
    following: list[DataBlock] = [second]
    manager = SessionManager(runner, "managed")
    stream = manager.events()
    events: list[SessionEvent] = []

    async def wait_for_terminal(run_id: str) -> None:
        """等待实际终态通知，允许工具结果 IO 线程完成调度。"""
        async with asyncio.timeout(2):
            while True:
                event = await anext(stream)
                events.append(event)
                if (
                    isinstance(event, RunEvent)
                    and event.run_id == run_id
                    and event.kind is RunEventKind.RUN_TERMINAL
                ):
                    return

    idle = await manager.submit(initial)
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    steer = await manager.submit(steering, mode="steer")
    follow_up = await manager.submit(following, mode="follow_up")
    assert idle.state == "delivered" and store.load_result(idle.run_id) is None
    assert steer.state == follow_up.state == "pending"
    assert store.load_run(follow_up.run_id) is None
    assert store.load_session("managed").messages[0].content == initial

    provider.release.set()
    await wait_for_terminal(follow_up.run_id)
    later = await manager.submit("再比较两张图片")
    assert later.state == "delivered"
    assert store.load_run(later.run_id).request.input == "再比较两张图片"
    await wait_for_terminal(later.run_id)
    await manager.close()
    events.extend([event async for event in stream])
    delivered = [
        event.submission_id
        for event in events
        if isinstance(event, SubmissionEvent) and event.state == "delivered"
    ]
    assert delivered == [steer.submission_id, follow_up.submission_id]
    for receipt, kind in (
        (steer, RunEventKind.TOOL_CALL_COMMITTED),
        (follow_up, RunEventKind.RUN_STARTED),
    ):
        delivery_index = next(
            index
            for index, event in enumerate(events)
            if isinstance(event, SubmissionEvent)
            and event.submission_id == receipt.submission_id
            and event.state == "delivered"
        )
        assert any(
            isinstance(event, RunEvent) and event.run_id == receipt.run_id and event.kind is kind
            for event in events[:delivery_index]
        )
    messages = store.load_session("managed").messages
    assert [
        message.content
        for message in messages
        if message.role.value == "user" and not message.tool_results
    ] == [
        initial,
        steering,
        following,
        "再比较两张图片",
    ]
    result = next(block for message in messages for block in message.tool_results)
    assert result.tool_use_id == "note" and result.text == "compare the two colors"
    assert len(provider.requests) == 4
    assert [
        message.content
        for message in provider.requests[1].messages
        if message.role.value == "user" and not message.tool_results
    ] == [
        initial,
        steering,
    ]
    for request in provider.requests[2:]:
        assert [
            block
            for message in request.messages
            for block in message.blocks
            if isinstance(block, ImageBlock)
        ] == [first, second, first, second]
    assert store.load_run(follow_up.run_id).request.input == following
    assert second.model.path.read_bytes() == _png("blue")


@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["before_input", "before_model"])
async def test_durable_request_and_committed_input_recover_without_original_source(
    tmp_path: Path,
    image_store: LifecycleStore,
    position: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """输入 commit 前后都能重开恢复，同一图片输入只进入原文一次。"""
    provider = BlockingProvider()
    runtime = build_runtime(tmp_path, provider=provider)
    runner = AgentRunner(runtime=runtime, store=image_store)
    source = tmp_path / "source.png"
    source.write_bytes(_png())
    image = await runner.import_image(source, session_id="restart", name="saved.png")
    entered = provider.started
    if position == "before_input":
        entered = asyncio.Event()

        async def block_before_input(*args: object, **kwargs: object) -> None:
            """模拟 create 已提交而输入 commit 尚未发生的进程中断。"""
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(runtime, "execute", block_before_input)
    running = asyncio.create_task(
        runner.start(AgentRunRequest(input=[image], session_id="restart", run_id="recover"))
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    source.write_bytes(b"source no longer contains an image")
    restarted = (
        SQLiteStore(image_store.path) if isinstance(image_store, SQLiteStore) else image_store
    )
    crashed = restarted.load_run("recover")
    checkpoint = restarted.load_checkpoint("recover")
    assert crashed.request.input == [image]
    assert checkpoint.engine_cursor["position"] == position
    assert "image-cache" in crashed.request.model_dump_json()
    assert "image-cache" not in checkpoint.model_dump_json()
    final_provider = StaticProvider(text_response())
    recovered_runtime = CountingAgentRuntime(build_runtime(tmp_path, provider=final_provider))
    result = await AgentRunner(runtime=recovered_runtime, store=restarted).recover(
        "recover", expected_activation_id=crashed.current_activation_id
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert recovered_runtime.activations[0].run_input == [image]
    assert final_provider.requests[0].messages[1].content == [image]
    assert restarted.load_session("restart").messages[0].content == [image]
    assert len(restarted.load_session("restart").messages) == 2
    assert image.model.path.read_bytes() == _png()


@pytest.mark.asyncio
async def test_hitl_resume_reuses_durable_image_input(tmp_path: Path) -> None:
    """人工答复只推进工具批次，不替换或再次提交原始图片输入。"""
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    store = SQLiteStore(tmp_path / "hitl.db")
    first = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(
                    ToolUseBlock(id="question", name="ask_question", input={"question": "继续？"})
                )
            ),
        ),
        store=store,
    )
    image = await first.import_image(_png(), session_id="default", name="question.png")
    waiting = await first.start(AgentRunRequest(input=[image], run_id="question-run"))
    runtime = CountingAgentRuntime(build_runtime(tmp_path, registry=registry))
    restarted = SQLiteStore(store.path)
    result = await AgentRunner(runtime=runtime, store=restarted).resume(
        "question-run",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="继续"),
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert runtime.activations[0].run_input == [image]
    assert (
        sum(message.content == [image] for message in restarted.load_session("default").messages)
        == 1
    )


@pytest.mark.asyncio
async def test_parent_resume_preserves_images_without_forwarding_to_child(tmp_path: Path) -> None:
    """Child 只接收明确的文字 prompt，parent 的恢复 activation 仍持有原图片。"""
    child = StaticProvider(
        tool_response(ToolUseBlock(id="ask", name="ask_question", input={"question": "继续？"})),
        text_response("Child done"),
    )
    parent = _parent_provider()
    runner = AgentRunner.from_config_path(
        _write_configs(tmp_path), provider=parent, child_provider_factory=ChildProviders(child)
    )
    runtime = CountingAgentRuntime(runner.runtime)
    runner.runtime = runtime
    image = await runner.import_image(_png(), session_id="default", name="parent.png")
    waiting = await runner.start(AgentRunRequest(input=[image], run_id="parent"))
    result = await runner.resume(
        "parent",
        interaction_id=waiting.pending_interaction.interaction_id,
        response=QuestionInteractionResponse(answer="继续"),
    )
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert [activation.kind for activation in runtime.activations] == ["start", "resume"]
    assert all(activation.run_input == [image] for activation in runtime.activations)
    assert any(message.content == [image] for message in parent.requests[-1].messages)
    assert any(message.text == "Child task" for message in child.requests[0].messages)
    assert not any(
        isinstance(block, ImageBlock)
        for request in child.requests
        for message in request.messages
        for block in message.blocks
    )
    await runner.aclose()
