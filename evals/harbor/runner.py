"""在任务容器内运行 Iris SDK；模型、工具和状态仍由 Iris 拥有。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from iris.agents import load_agent_config
from iris.config import init_config
from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunLimits,
    RunPhase,
    RunResult,
    RunStopReason,
)
from iris.providers.factory import parse_model_route
from iris.store import SQLiteStore

from .results import save_results, write_json

logger = logging.getLogger(__name__)


async def _settle(runner: AgentRunner, task: asyncio.Task[RunResult], run_id: str) -> None:
    """按准入状态停止原调用，再结算可能留下的 WAITING。"""
    run = runner.store.load_run(run_id)
    if not task.done():
        if run is None:
            task.cancel()
        else:
            runner.request_cancel(run_id, reason="Harbor stopped Iris")
            if run.phase is RunPhase.WAITING:
                task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    run = runner.store.load_run(run_id)
    if run is not None and run.phase is RunPhase.WAITING:
        await runner.cancel(run_id, reason="Harbor does not answer interactions")


async def run_agent(
    *,
    profile: Path,
    model: str,
    instruction: str,
    workspace: Path,
    logs_dir: Path,
    max_model_steps: int,
    stop: asyncio.Event,
) -> int:
    """执行一次终端任务，返回 0 正常、1 运行失败、70 资源收尾失败。

    Args:
        profile: 容器内自包含 Agent YAML。
        model: Harbor 指定的 provider/model 主模型路由。
        instruction: 原始任务指令。
        workspace: 任务环境的实际默认工作目录。
        logs_dir: 本 invocation 的持久日志目录。
        max_model_steps: Iris 自身模型步数上限。
        stop: SIGTERM 等宿主停止请求的事件。

    Returns:
        int: 原业务与资源收尾决定的退出码，归档状态单独记录。
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    session_id, run_id = f"harbor_{uuid4().hex}", f"run_{uuid4().hex}"
    write_json(
        logs_dir / "control.json",
        {
            "pid": os.getpid(),
            "session_id": session_id,
            "run_id": run_id,
        },
    )
    runner: AgentRunner | None = None
    task: asyncio.Task[RunResult] | None = None
    stopping: asyncio.Task[bool] | None = None
    result: RunResult | None = None
    execution_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        route = parse_model_route(model)
        config = load_agent_config(profile)
        config = config.model_copy(
            update={
                "model": config.model.model_copy(
                    update={"provider": route.provider, "name": route.model}
                ),
                "permissions": config.permissions.model_copy(
                    update={"workspace": str(workspace.resolve())}
                ),
                "prompts": config.prompts.model_copy(
                    update={"root": str(profile.parent / "runtime-prompts")}
                ),
            }
        )
        runner = AgentRunner.from_config(
            config, config_path=profile, store=SQLiteStore(logs_dir / "lifecycle.db")
        )
        task = asyncio.create_task(
            runner.start(
                AgentRunRequest(input=instruction, session_id=session_id, run_id=run_id),
                options=AgentRunOptions(limits=RunLimits(max_model_steps=max_model_steps)),
            )
        )
        stopping = asyncio.create_task(stop.wait())
        done, _ = await asyncio.wait({task, stopping}, return_when=asyncio.FIRST_COMPLETED)
        if stopping in done and not task.done():
            await _settle(runner, task, run_id)
        with suppress(asyncio.CancelledError):
            result = await task
    except (Exception, asyncio.CancelledError) as error:
        execution_error = error
        logger.exception("Iris 容器调用中断或失败")
    finally:
        if stopping is not None:
            stopping.cancel()
            with suppress(asyncio.CancelledError):
                await stopping
        if runner is not None:
            try:
                try:
                    if task is not None:
                        await _settle(runner, task, run_id)
                finally:
                    await runner.aclose()
            except (Exception, asyncio.CancelledError) as error:
                cleanup_error = error
                logger.exception("Iris 资源收尾失败")
            if result is None:
                result = runner.store.load_result(run_id)
        save_results(
            runner=runner,
            session_id=session_id,
            run_id=run_id,
            model=model,
            logs_dir=logs_dir,
            result=result,
            cleanup_error=cleanup_error,
            execution_error=execution_error,
        )
    if cleanup_error is not None:
        return 70
    if execution_error is not None or result is None:
        return 1
    return (
        0
        if result.run.stop_reason in {RunStopReason.COMPLETED, RunStopReason.BUDGET_EXHAUSTED}
        else 1
    )


async def _main(arguments: argparse.Namespace) -> int:
    """Linux 容器入口把 SIGTERM 转为 SDK 停止请求。"""
    init_config()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    try:
        return await run_agent(
            profile=arguments.profile.resolve(),
            model=arguments.model,
            instruction=arguments.instruction_file.read_text(encoding="utf-8"),
            workspace=arguments.workspace.resolve(),
            logs_dir=arguments.logs_dir.resolve(),
            max_model_steps=arguments.max_model_steps,
            stop=stop,
        )
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main() -> None:
    """解析非交互入口参数并使用标准 logging 输出错误。"""
    parser = argparse.ArgumentParser(description="Run Iris inside a Harbor task environment")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--max-model-steps", type=int, default=20)
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(asyncio.run(_main(parser.parse_args())))


if __name__ == "__main__":
    main()
