"""由真实 Runner 运行 Python 代码片段并保留工作区报告，无需 API key。"""

import argparse
import asyncio
import json
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
    """只注册 run_python，执行标准库计算并从工作区读取生成的报告。"""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    provider = ScriptedProvider()
    provider.steps.extend(
        [
            tool(
                "python-report",
                "run_python",
                code=(
                    "import json\n"
                    "from pathlib import Path\n"
                    "values = [3, 4, 5]\n"
                    "report = {'rows': len(values), 'total': sum(values)}\n"
                    "Path('report.json').write_text(json.dumps(report), encoding='utf-8')\n"
                    "print(json.dumps(report))\n"
                ),
            ),
            done("Python 已完成计算并将报告写入工作区。"),
        ]
    )
    runner = AgentRunner.from_config(
        config_for_workspace("python.yaml", workspace), provider=provider
    )
    confirmed: list[str] = []
    try:
        result = await runner.start(
            AgentRunRequest(input="计算数据汇总并生成报告。", session_id="python")
        )
        result = await approve_commands(runner, result, confirmed)
        require_completed(runner, result)
        calls = runner.list_tool_calls(result.run.run_id)
        return {
            "workspace": str(workspace),
            "tool_names": [call.tool_name for call in calls],
            "confirmed_tools": confirmed,
            "output": json.loads((workspace / "report.json").read_text(encoding="utf-8")),
            "model_result": calls[0].result.model_content,
        }
    finally:
        await runner.aclose()


def main() -> None:
    """从命令行运行，默认建立本次独立的工作区。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    workspace = args.workspace or Path("tmp") / f"python-{uuid4().hex[:8]} 中文 workspace"
    print(json.dumps(asyncio.run(run_example(workspace)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
