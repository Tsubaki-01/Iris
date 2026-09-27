"""用真实 SDK 和确定性 provider 验证 Inspect 接入，不运行 benchmark。"""

# ruff: noqa: E402
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("inspect_ai")

import anyio
from inspect_ai import Task, eval_async
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ModelName
from inspect_ai.solver import TaskState

from evals.iris_solver import IrisSample, iris_solver
from iris.agents import AgentConfig
from iris.exceptions import IrisConfigError, IrisProviderError, IrisRunStateError
from iris.harness import AgentRunner
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import AgentRunOptions, LifecycleStore, RunLimits, RunResult, RunStopReason
from iris.message import LLMRequest, LLMResponse, TextBlock, ToolUseBlock


class FakeProvider:
    """仅实现非流式协议，按顺序返回响应或异常。"""

    def __init__(self, *responses: LLMResponse | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """返回受控估算，不调用外部模型。"""
        return 1

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """记录请求，消费一项受控响应。"""
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class BlockingProvider(FakeProvider):
    """暴露调用同步点，用于验证外部取消。"""

    def __init__(self, *responses: LLMResponse) -> None:
        super().__init__(*responses)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """等到宿主取消本次调用。"""
        if self.responses:
            return await super().complete(request)
        self.started.set()
        await self.release.wait()
        return response("测试主动释放")


def response(text: str) -> LLMResponse:
    """构造带真实计数结构的受控响应。"""
    return LLMResponse(
        provider="fake",
        model="fake-model",
        content=[TextBlock(text=text)],
        finish_reason="stop",
        input_tokens=3,
        output_tokens=2,
        total_tokens=5,
    )


def question_response() -> LLMResponse:
    """让真实工具执行链进入 WAITING。"""
    return LLMResponse(
        provider="fake",
        model="fake-model",
        content=[ToolUseBlock(id="ask-1", name="ask_question", input={"question": "选哪个？"})],
        finish_reason="tool_calls",
        input_tokens=4,
        output_tokens=1,
        total_tokens=5,
    )


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """提供不含长期记忆或外部工具的最小 Agent 配置。"""
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: eval-test\nmodel: openai/fake-model\nsystem: 完成输入。\n"
        "tools:\n  builtin: [human.ask]\n",
        encoding="utf-8",
    )
    return path


def install_providers(
    monkeypatch: pytest.MonkeyPatch,
    *providers: FakeProvider,
    child_provider: FakeProvider | None = None,
) -> list[AgentRunner]:
    """保留真实配置装配，只替换外部 provider。"""
    original = AgentRunner.from_config_path
    pending = list(providers)
    runners: list[AgentRunner] = []

    def child_factory(config: AgentConfig, *, config_path: Path) -> FakeProvider | None:
        return child_provider

    def create(cls: type[AgentRunner], path: str | Path, *, store: LifecycleStore) -> AgentRunner:
        runner = original(
            path,
            provider=pending.pop(0),
            store=store,
            child_provider_factory=child_factory if child_provider is not None else None,
        )
        runners.append(runner)
        return runner

    monkeypatch.setattr(AgentRunner, "from_config_path", classmethod(create))
    return runners


def task_state(prompt: str = "受控输入") -> TaskState:
    """直接构造 Inspect 状态，不加载任何真实题集。"""
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id="fixture",
        epoch=1,
        input=prompt,
        messages=[ChatMessageUser(content=prompt)],
        metadata={"fixture": True},
    )


async def no_generate(state: TaskState, **kwargs: Any) -> TaskState:
    """Inspect 的模型入口不应参与 Iris 执行。"""
    pytest.fail("接入层不能调用 Inspect generate")


@pytest.mark.asyncio
async def test_text_input_result_and_usage(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(response("完成"))
    runners = install_providers(monkeypatch, provider)
    state = task_state()
    state.messages[0].content = "前置 solver 修改后的输入"

    result = await iris_solver(str(config_path))(state, no_generate)

    assert result is state
    assert state.output.completion == "完成"
    assert state.output.model == "openai/fake-model"
    assert state.messages[-1].text == "完成"
    assert state.completed
    assert state.metadata == {"fixture": True}
    record = state.store.get("iris")
    assert record["runs"][0]["stop_reason"] == "completed"
    assert record["runs"][0]["usage"]["total_tokens"] == 5
    assert record["runs"][0]["usage"]["compaction"]["total_tokens"] == 0
    assert "success" not in record
    assert state.output.usage is None
    assert not provider.requests[0].stream
    assert provider.requests[0].messages[-1].text == "前置 solver 修改后的输入"
    with pytest.raises(IrisRunStateError, match="关闭"):
        await runners[0].aprepare()


@pytest.mark.asyncio
async def test_concurrent_samples_have_distinct_sessions(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runners = install_providers(
        monkeypatch, FakeProvider(response("一")), FakeProvider(response("二"))
    )
    solve = iris_solver(str(config_path))
    states = await asyncio.gather(
        solve(task_state("甲"), no_generate), solve(task_state("乙"), no_generate)
    )

    records = [state.store.get("iris") for state in states]
    assert records[0]["session_id"] != records[1]["session_id"]
    assert runners[0].store is not runners[1].store
    assert [state.output.completion for state in states] == ["一", "二"]
    for runner, record, prompt in zip(runners, records, ("甲", "乙"), strict=True):
        assert runner.get_session(record["session_id"]).messages[0].text == prompt


@pytest.mark.asyncio
async def test_waiting_is_recorded_then_cancelled_without_answer(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(question_response())
    runners = install_providers(monkeypatch, provider)

    state = await iris_solver(str(config_path))(task_state(), no_generate)

    record = state.store.get("iris")["runs"][0]
    assert record["phase"] == "waiting"
    assert record["pending_interaction"] is not None
    assert record["stop_reason"] is None
    assert len(provider.requests) == 1
    assert runners[0].get_result(record["run_id"]).run.stop_reason is RunStopReason.CANCELLED


@pytest.mark.asyncio
async def test_callback_can_resume_and_start_another_turn(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(question_response(), response("第一轮"), response("第二轮"))
    runners = install_providers(monkeypatch, provider)

    async def execute(sample: IrisSample, state: TaskState) -> RunResult:
        waiting = await sample.start(state.user_prompt.text)
        resumed = await sample.resume(waiting, QuestionInteractionResponse(answer="选甲"))
        assert resumed.assistant_message.text == "第一轮"
        return await sample.start("继续")

    state = await iris_solver(str(config_path), execute=execute)(task_state(), no_generate)

    records = state.store.get("iris")["runs"]
    assert state.output.completion == "第二轮"
    assert len(records) == 2
    assert [record["usage"]["total_tokens"] for record in records] == [10, 5]
    assert all(record["stop_reason"] == "completed" for record in records)
    history = runners[0].get_session(state.store.get("iris")["session_id"]).messages
    assert history[-2].text == "继续"


@pytest.mark.asyncio
async def test_provider_failure_preserves_run_error(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_providers(monkeypatch, FakeProvider(IrisProviderError("受控 provider 失败")))
    state = await iris_solver(str(config_path))(task_state(), no_generate)

    record = state.store.get("iris")["runs"][0]
    assert record["stop_reason"] == "failed"
    assert record["error"]["source"] == "provider"
    assert state.output.completion == ""


@pytest.mark.asyncio
async def test_callback_exception_cleans_waiting_run(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runners = install_providers(monkeypatch, FakeProvider(question_response()))
    run_ids: list[str] = []

    async def execute(sample: IrisSample, state: TaskState) -> RunResult:
        result = await sample.start("提问")
        run_ids.append(result.run.run_id)
        raise LookupError("受控任务回调异常")

    with pytest.raises(LookupError, match="任务回调"):
        await iris_solver(str(config_path), execute=execute)(task_state(), no_generate)

    assert runners[0].get_result(run_ids[0]).run.stop_reason is RunStopReason.CANCELLED
    with pytest.raises(IrisRunStateError, match="关闭"):
        await runners[0].aprepare()


@pytest.mark.asyncio
async def test_external_cancellation_settles_before_close(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = BlockingProvider()
    runners = install_providers(monkeypatch, provider)
    samples: list[IrisSample] = []

    async def execute(sample: IrisSample, state: TaskState) -> RunResult:
        samples.append(sample)
        return await sample.start("等待取消")

    running = asyncio.create_task(
        iris_solver(str(config_path), execute=execute)(task_state(), no_generate)
    )
    await asyncio.wait_for(provider.started.wait(), timeout=3)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    run = samples[0].results()[0]
    assert run.run.stop_reason is RunStopReason.CANCELLED
    with pytest.raises(IrisRunStateError, match="关闭"):
        await runners[0].aprepare()


@pytest.mark.asyncio
async def test_run_options_are_forwarded(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        LLMResponse(
            provider="fake",
            model="fake-model",
            content=[ToolUseBlock(id="missing", name="missing_tool", input={})],
            finish_reason="tool_calls",
        )
    )
    install_providers(monkeypatch, provider)

    async def execute(sample: IrisSample, state: TaskState) -> RunResult:
        result = await sample.start(
            "输入", options=AgentRunOptions(limits=RunLimits(max_model_steps=1))
        )
        assert result.run.limits.max_model_steps == 1
        return result

    state = await iris_solver(str(config_path), execute=execute)(task_state(), no_generate)
    assert state.store.get("iris")["runs"][0]["stop_reason"] == "budget_exhausted"


@pytest.mark.asyncio
async def test_cancel_before_run_admission(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider()
    runners = install_providers(monkeypatch, provider)
    preparing = asyncio.Event()
    prepare = AgentRunner.aprepare

    async def blocked_prepare(self: AgentRunner) -> None:
        preparing.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(AgentRunner, "aprepare", blocked_prepare)
    running = asyncio.create_task(iris_solver(str(config_path))(task_state(), no_generate))
    await asyncio.wait_for(preparing.wait(), timeout=3)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert provider.requests == []
    with pytest.raises(IrisRunStateError, match="关闭"):
        await prepare(runners[0])


@pytest.mark.asyncio
async def test_anyio_scope_cancellation_drains_sdk(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspect 使用 AnyIO scope，收尾不能被同一次取消反复打断。"""
    provider = BlockingProvider()
    runners = install_providers(monkeypatch, provider)
    samples: list[IrisSample] = []
    close = AgentRunner.aclose

    async def delayed_close(self: AgentRunner) -> None:
        # MCP 等资源关闭会让出控制权，不能只测无异步清理的空环境。
        await anyio.sleep(0)
        await close(self)

    monkeypatch.setattr(AgentRunner, "aclose", delayed_close)

    async def execute(sample: IrisSample, state: TaskState) -> RunResult:
        samples.append(sample)
        return await sample.start("等待取消")

    async def run() -> None:
        await iris_solver(str(config_path), execute=execute)(task_state(), no_generate)

    with anyio.fail_after(3):
        async with anyio.create_task_group() as group:
            group.start_soon(run)
            await provider.started.wait()
            group.cancel_scope.cancel()

    assert samples[0].results()[0].run.stop_reason is RunStopReason.CANCELLED
    with pytest.raises(IrisRunStateError, match="关闭"):
        await runners[0].aprepare()


@pytest.mark.asyncio
async def test_cancel_while_resuming_child_proxy(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path.write_text(
        "name: parent\nmodel: openai/fake-model\nsystem: 委派任务。\n"
        "tools:\n  subagent: subagents.yaml\n",
        encoding="utf-8",
    )
    (config_path.parent / "subagents.yaml").write_text(
        "default: child\nagents:\n  child:\n    path: child.yaml\n    description: 处理子任务\n",
        encoding="utf-8",
    )
    (config_path.parent / "child.yaml").write_text(
        "name: child\nmodel: openai/fake-model\nsystem: 先提问再执行。\n"
        "tools:\n  builtin: [human.ask]\n",
        encoding="utf-8",
    )
    parent = FakeProvider(
        LLMResponse(
            provider="fake",
            model="fake-model",
            content=[ToolUseBlock(id="delegate", name="subagent", input={"prompt": "子任务"})],
            finish_reason="tool_calls",
        ),
        response("父任务结束"),
    )
    child = BlockingProvider(question_response())
    runners = install_providers(monkeypatch, parent, child_provider=child)
    run_ids: list[str] = []

    async def execute(sample: IrisSample, state: TaskState) -> RunResult:
        waiting = await sample.start("开始")
        run_ids.append(waiting.run.run_id)
        return await sample.resume(waiting, QuestionInteractionResponse(answer="继续"))

    running = asyncio.create_task(
        iris_solver(str(config_path), execute=execute)(task_state(), no_generate)
    )
    await asyncio.wait_for(child.started.wait(), timeout=3)
    running.cancel()
    done, _ = await asyncio.wait({running}, timeout=2)
    if not done:
        child.release.set()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert done, "取消不应等待 child provider 自行完成"
    assert runners[0].get_result(run_ids[0]).run.stop_reason is RunStopReason.CANCELLED


@pytest.mark.asyncio
async def test_default_rejects_conversation_instead_of_dropping_messages(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_providers(monkeypatch, FakeProvider())
    state = task_state()
    state.messages.append(ChatMessageUser(content="第二条输入"))

    with pytest.raises(IrisConfigError, match="execute"):
        await iris_solver(str(config_path))(state, no_generate)


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", [False, True])
async def test_inspect_dispatch_and_log_with_fake_provider(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, custom: bool
) -> None:
    install_providers(monkeypatch, FakeProvider(response("接口结果")))

    async def execute(sample: IrisSample, state: TaskState) -> RunResult:
        return await sample.start(state.user_prompt.text)

    logs = await eval_async(
        Task(
            dataset=[Sample(input="接口测试 fixture", id="fixture")],
            solver=iris_solver(str(config_path), execute=execute if custom else None),
        ),
        model="mockllm/model",
        score=False,
        log_dir=str(tmp_path / "inspect-logs"),
    )

    assert logs[0].status == "success"
    assert logs[0].samples[0].output.completion == "接口结果"
    assert logs[0].samples[0].store["iris"]["runs"][0]["usage"]["total_tokens"] == 5
