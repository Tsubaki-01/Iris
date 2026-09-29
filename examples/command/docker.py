"""显式启用的真实 Docker 跨 session 示例；仅消费预备镜像。"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from iris.exceptions import IrisProviderError
from iris.harness import AgentRunner
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import AgentRunRequest, RunPhase

from ._scripted import (
    ScriptedProvider,
    approve_commands,
    config_for_workspace,
    done,
    require_completed,
    tool,
)


async def run_example(workspace: Path, *, image: str | None = None) -> dict[str, Any]:
    """两个 session 共用一个 root，正常结果与命令均经 Runner 持久化。"""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    config = config_for_workspace("docker.yaml", workspace)
    if image is not None:
        config = config.model_copy(
            update={
                "command": config.command.model_copy(
                    update={"docker": config.command.docker.model_copy(update={"image": image})}
                )
            }
        )
    provider = ScriptedProvider()
    runner = AgentRunner.from_config(config, provider=provider)
    confirmed: list[str] = []
    try:
        # Docker 显式 prepare；它验证已有引擎/镜像，不 pull 或 build。
        await runner.aprepare()
        provider.steps.extend(
            [
                tool(
                    "write-module",
                    "write_file",
                    file_path="offline_stats.py",
                    content=Path(__file__)
                    .with_name("offline_stats.py")
                    .read_text(encoding="utf-8"),
                ),
                tool(
                    "write-service",
                    "write_file",
                    file_path="service_demo.py",
                    content=Path(__file__).with_name("service_demo.py").read_text(encoding="utf-8"),
                ),
                tool(
                    "start-service", "exec_command", command="python service_demo.py start a.json"
                ),
                done("A 已启动服务；正常完成保留它。"),
            ]
        )
        first = await runner.start(AgentRunRequest(input="启动离线服务。", session_id="a"))
        first = await approve_commands(runner, first, confirmed)
        require_completed(runner, first)

        provider.steps.extend(
            [
                tool(
                    "reuse-service", "exec_command", command="python service_demo.py check b.json"
                ),
                tool("read-b", "read_file", file_path="b.json"),
                tool("wait-b", "ask_question", question="服务已复用，是否继续验证整体停止？"),
            ]
        )
        waiting = await runner.start(AgentRunRequest(input="复用 A 的服务后等待。", session_id="b"))
        waiting = await approve_commands(runner, waiting, confirmed)
        assert waiting.run.phase is RunPhase.WAITING

        # 明确模拟 run 失败；非零命令返回只是工具错误，不能冒充 run 失败。
        provider.steps.append(IrisProviderError("示例模拟 provider 失败，触发 run 异常结算。"))
        failed = await runner.start(
            AgentRunRequest(input="模拟 A 的下一次任务失败。", session_id="a")
        )
        waiting_survived = runner.get_run(waiting.run.run_id).phase is RunPhase.WAITING

        provider.steps.extend(
            [
                tool(
                    "verify-stopped",
                    "exec_command",
                    command="python service_demo.py stopped stopped.json",
                ),
                tool("read-stopped", "read_file", file_path="stopped.json"),
                tool(
                    "restart-service",
                    "exec_command",
                    command="python service_demo.py start restart.json",
                ),
                tool(
                    "reuse-restarted",
                    "exec_command",
                    command="python service_demo.py check after.json",
                ),
                tool("read-after", "read_file", file_path="after.json"),
                done("B 的 HITL 继续；文件保留，后台服务已经显式重启。"),
            ]
        )
        result = await runner.resume(
            waiting.run.run_id,
            interaction_id=waiting.pending_interaction.interaction_id,
            response=QuestionInteractionResponse(answer="继续"),
        )
        result = await approve_commands(runner, result, confirmed)
        require_completed(runner, result)
        return {
            "workspace": str(workspace),
            "shell": runner.runtime.environment.command_environment.command_shell,
            "session_a": json.loads((workspace / "a.json").read_text(encoding="utf-8")),
            "session_b": json.loads((workspace / "b.json").read_text(encoding="utf-8")),
            "waiting_survived": waiting_survived,
            "failure_reason": failed.run.stop_reason.value,
            "after_stop": json.loads((workspace / "stopped.json").read_text(encoding="utf-8")),
            "after_restart": json.loads((workspace / "after.json").read_text(encoding="utf-8")),
            "confirmed_tools": confirmed,
        }
    finally:
        # root 关闭会删除本次容器；/workspace 中的宿主文件保留。
        await runner.aclose()


def main() -> None:
    """命令行必须显式启用 Docker；默认不连接引擎。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-docker", action="store_true", help="显式连接本地 Linux engine")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--image", help="已显式构建的本地镜像；默认读取 docker.yaml")
    args = parser.parse_args()
    if not args.run_docker:
        parser.error("请显式构建本地镜像后传入 --run-docker；示例不自动构建或拉取镜像")
    workspace = args.workspace or Path("tmp") / f"docker-{uuid4().hex[:8]} 中文 workspace"
    print(
        json.dumps(
            asyncio.run(run_example(workspace, image=args.image)), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
