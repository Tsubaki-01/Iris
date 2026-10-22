"""把 Inspect sample 接入 Iris SDK，不加载题集或定义评分规则。"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from anyio import CancelScope
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver

from iris.exceptions import IrisConfigError
from iris.harness import AgentRunner, collect_trajectory, write_trajectory
from iris.hitl import HumanInteractionResponse
from iris.lifecycle import AgentRunOptions, AgentRunRequest, RunPhase, RunResult
from iris.store import InMemoryLifecycleStore

logger = logging.getLogger(__name__)


class IrisSample:
    """拥有一次 sample 的 runner 资源与 run 引用，不持有第二份运行状态。

    回调通过 start/resume 顺序执行任务，通过 runner 查询历史和工具产物。
    runner 由 solver 创建和关闭，回调不另开后台运行任务。
    """

    def __init__(self, runner: AgentRunner) -> None:
        """绑定本 sample 独占的 runner，并创建独立 session ID。"""
        self.runner = runner
        self.session_id = f"eval_{uuid4().hex}"
        self._run_ids: list[str] = []

    async def start(self, input: str, *, options: AgentRunOptions | None = None) -> RunResult:
        """在同一 sample session 中开始一轮输入。

        Args:
            input: 本轮交给 Iris 的用户文本。
            options: 直接传给 AgentRunner 的预算和执行选项。

        Returns:
            RunResult: SDK 的 waiting 或 terminal 结果。
        """
        run_id = f"run_{uuid4().hex}"
        self._run_ids.append(run_id)
        request = AgentRunRequest(input=input, session_id=self.session_id, run_id=run_id)
        return await self._invoke(self.runner.start(request, options=options), run_id)

    async def resume(self, result: RunResult, response: HumanInteractionResponse) -> RunResult:
        """用具体任务协议给出的 typed response 继续一次 WAITING。

        Args:
            result: 此 sample 中需要继续的 SDK 结果。
            response: 任务回调决定的回答或权限响应。

        Returns:
            RunResult: 同一 run 更新后的累计结果。
        """
        return await self._invoke(
            self.runner.resume(
                result.run.run_id,
                interaction_id=result.run.pending_interaction_id,
                response=response,
            ),
            result.run.run_id,
        )

    def results(self) -> list[RunResult]:
        """按创建顺序读取已产生的结果；resume 不重复累计同一 run。"""
        return [
            result
            for run_id in self._run_ids
            if (result := self.runner.store.load_result(run_id)) is not None
        ]

    async def _invoke(self, operation: Coroutine[Any, Any, RunResult], run_id: str) -> RunResult:
        """取消先通知 durable owner，再等待原 SDK 调用完成清理。"""
        task = asyncio.create_task(operation)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Inspect 的 AnyIO deadline 会反复取消 await，收尾必须屏蔽该次取消。
            with CancelScope(shield=True):
                # prepare 尚未创建 run 时只能取消原任务。
                if self.runner.store.load_run(run_id) is None:
                    task.cancel()
                else:
                    run = self.runner.request_cancel(run_id, reason="Inspect sample cancelled")
                    # child proxy 的 resume 尚处于 WAITING，需中断原调用触发 child 结算。
                    if run.phase is RunPhase.WAITING:
                        task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            raise

    async def _close(self) -> None:
        """结算回调留下的 WAITING，再关闭本 sample 的运行资源。"""
        try:
            for run_id in self._run_ids:
                run = self.runner.store.load_run(run_id)
                if run is not None and run.phase is RunPhase.WAITING:
                    await self.runner.cancel(run_id, reason="Inspect sample finished")
        finally:
            await self.runner.aclose()


SampleExecutor = Callable[[IrisSample, TaskState], Awaitable[RunResult]]


async def _execute_text(sample: IrisSample, state: TaskState) -> RunResult:
    """承接当前单条用户文本，复杂输入由具体任务的 execute 回调解释。"""
    if (
        len(state.messages) != 1
        or not isinstance(state.messages[0], ChatMessageUser)
        or not isinstance(state.messages[0].content, str)
    ):
        raise IrisConfigError("默认执行只接受一条用户文本；对话或多模态输入请提供 execute 回调")
    return await sample.start(state.messages[0].content)


def _run_record(result: RunResult) -> dict[str, Any]:
    """导出 SDK 原始状态和用量，不把执行结束解释成评分成功。"""
    return {
        "run_id": result.run.run_id,
        "phase": result.run.phase.value,
        "stop_reason": result.run.stop_reason.value if result.run.stop_reason is not None else None,
        "usage": result.run.usage.model_dump(mode="json"),
        "error": result.error.model_dump(mode="json") if result.error is not None else None,
        "pending_interaction": (
            result.pending_interaction.model_dump(mode="json")
            if result.pending_interaction is not None
            else None
        ),
    }


def _export_sample(
    sample: IrisSample,
    state: TaskState,
    directory: Path,
    cleanup_error: BaseException | None,
) -> None:
    """记录已收尾或部分可读的轨迹，导出错误不改变业务执行结果。"""
    try:
        document = collect_trajectory(sample.runner, sample.session_id)
        document.update(
            {
                "archive_id": state.uuid,
                "archive_root": str(directory),
                "sample_id": state.sample_id,
                "epoch": state.epoch,
            }
        )
        if cleanup_error is not None:
            document["export"]["status"] = "partial"
            document["export"]["errors"].append(
                {
                    "stage": "cleanup",
                    "message": str(cleanup_error),
                }
            )
        exported = write_trajectory(document, directory / state.uuid)
    except Exception as error:
        # 归档是宿主观察出口；不能用其异常替换原运行或清理异常。
        logger.exception("Inspect 轨迹导出失败")
        exported = {
            "root_session_id": sample.session_id,
            "archive_id": state.uuid,
            "export": {"status": "failed", "errors": [{"stage": "collect", "message": str(error)}]},
        }
    state.store.set("iris_trace", exported)


@solver(name="iris")
def iris_solver(
    config_path: str, *, trajectory_dir: str | Path, execute: SampleExecutor | None = None
) -> Solver:
    """创建使用 Iris 自身 provider 的 Inspect Solver。

    Args:
        config_path: Agent YAML 路径，相对路径在创建 solver 时解析。
        trajectory_dir: 本次评测的归档根目录，每个 sample 按 state.uuid 独立保存。
        execute: 可选任务回调，用 IrisSample.start/resume 驱动多轮和 WAITING，
            返回作为最终输出的 RunResult。默认执行当前单条用户文本。

    Returns:
        Solver: 每个 sample 创建独立 runner 和进程内 lifecycle store 的异步入口。

    Note:
        workspace、memory 和外部工具环境沿用 YAML，不自动隔离。Inspect generate、
        模型配置及 token/cost 限额不接管 Iris；运行限额通过 AgentRunOptions 传递。
    """
    path = Path(config_path).resolve()
    directory = Path(trajectory_dir).resolve()
    execute_sample = execute if execute is not None else _execute_text

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        runner = AgentRunner.from_config_path(path, store=InMemoryLifecycleStore())
        sample = IrisSample(runner)
        try:
            result = await execute_sample(sample, state)
            # 收尾取消 WAITING 前保留任务原始结果，避免把取消误当作模型的停止原因。
            state.store.set(
                "iris",
                {
                    "session_id": sample.session_id,
                    "output_run_id": result.run.run_id,
                    "runs": [_run_record(item) for item in sample.results()],
                },
            )
            model = runner.runtime.environment.agent_config.model
            text = result.assistant_message.text if result.assistant_message is not None else ""
            state.output = ModelOutput.from_content(f"{model.provider}/{model.name}", text)
            if result.assistant_message is not None:
                state.messages.append(state.output.message)
            state.completed = True
            return state
        finally:
            operation_error = sys.exception()
            cleanup_error: BaseException | None = None
            with CancelScope(shield=True):
                try:
                    await sample._close()
                except BaseException as error:
                    cleanup_error = error
                    if operation_error is None:
                        raise
                    logger.exception("Inspect 资源收尾失败，保留原调用异常")
                finally:
                    _export_sample(sample, state, directory, cleanup_error)

    return solve


__all__ = ["IrisSample", "SampleExecutor", "iris_solver"]
