"""真实 Native 文件写入、命令处理与文件读取；无需 API key。"""

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest

from ._scripted import (
    ScriptedProvider,
    approve_commands,
    config_for_workspace,
    done,
    require_completed,
    tool,
)


async def run_example(workspace: Path) -> dict[str, Any]:
    """在新 workspace 中通过真实 Runner 执行固定工具链。"""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    provider = ScriptedProvider()
    arguments = [sys.executable, "transform.py"]
    command = subprocess.list2cmdline(arguments) if os.name == "nt" else shlex.join(arguments)
    provider.steps.extend(
        [
            tool("write-csv", "write_file", file_path="input.csv", content="value\n3\n4\n5\n"),
            tool(
                "write-script",
                "write_file",
                file_path="transform.py",
                content=Path(__file__).with_name("transform.py").read_text(encoding="utf-8"),
            ),
            tool("transform", "exec_command", command=command),
            tool("read-json", "read_file", file_path="output.json"),
            done("CSV 已经通过当前平台 shell 转换为 JSON，并由原生文件工具读回。"),
        ]
    )
    runner = AgentRunner.from_config(
        config_for_workspace("native.yaml", workspace), provider=provider
    )
    confirmed: list[str] = []
    try:
        result = await runner.start(
            AgentRunRequest(input="转换 CSV 并读回 JSON。", session_id="native")
        )
        result = await approve_commands(runner, result, confirmed)
        require_completed(runner, result)
        calls = runner.list_tool_calls(result.run.run_id)
        return {
            "workspace": str(workspace),
            "shell": runner.runtime.environment.command_environment.command_shell,
            "tool_names": [call.tool_name for call in calls],
            "confirmed_tools": confirmed,
            "output": json.loads((workspace / "output.json").read_text(encoding="utf-8")),
            "file_tool_read": calls[-1].result.model_content,
        }
    finally:
        await runner.aclose()


def main() -> None:
    """从命令行运行；默认创建本次独立的可保留 workspace。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    workspace = args.workspace or Path("tmp") / f"native-{uuid4().hex[:8]} 中文 workspace"
    print(json.dumps(asyncio.run(run_example(workspace)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
