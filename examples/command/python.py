"""由真实 Runner 执行 Python、回传错误并发布报告副本，无需 API key。"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from iris.exceptions import IrisRunStateError, IrisToolExecutionError
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest, RunStopReason
from iris.message import LLMRequest, LLMResponse

from ._scripted import (
    ScriptedProvider,
    approve_commands,
    config_for_workspace,
    done,
    tool,
)


class ReportProvider(ScriptedProvider):
    """返回固定响应，同时记录提交修正代码前真正收到的工具错误。"""

    def __init__(self) -> None:
        """初始化脚本及本次错误反馈观测。"""
        super().__init__()
        self.error_feedback = ""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """从修正代码对应的请求中提取前次执行错误，保留正常 provider 入口。"""
        response = await super().complete(request)
        if any(call.id == "python-report" for call in response.to_msg().tool_calls):
            self.error_feedback = "\n".join(
                result.content
                for message in request.messages
                for result in message.tool_results
                if result.tool_use_id == "python-broken" and result.is_error
            )
        return response


async def run_example(workspace: Path) -> dict[str, Any]:
    """依次准备 CSV、执行错误/修正代码，最后从已发布副本读取报告。"""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    provider = ReportProvider()
    report_code = (
        "import csv, json\n"
        "from pathlib import Path\n"
        "with open('input.csv', encoding='utf-8', newline='') as source:\n"
        "    values = [int(row['value']) for row in csv.DictReader(source)]\n"
        "report = {'rows': len(values), 'total': sum(values)}\n"
        "Path('report.json').write_text(json.dumps(report), encoding='utf-8')\n"
        "print(json.dumps(report))\n"
    )
    provider.steps.extend(
        [
            tool("write-csv", "write_file", file_path="input.csv", content="value\n3\n4\n5\n"),
            tool(
                "python-broken",
                "run_python",
                code=report_code.replace("row['value']", "row['missing']"),
            ),
            tool(
                "python-report",
                "run_python",
                code=report_code,
            ),
            tool("publish-report", "publish_artifact", file_path="report.json"),
            done("Python 报错已反馈，修正后的报告已发布为本地副本。"),
        ]
    )
    runner = AgentRunner.from_config(
        config_for_workspace("python.yaml", workspace), provider=provider
    )
    confirmed: list[str] = []
    try:
        result = await runner.start(
            AgentRunRequest(input="处理 CSV 并交付汇总报告。", session_id="python")
        )
        result = await approve_commands(runner, result, confirmed)
        if result.run.stop_reason is not RunStopReason.COMPLETED:
            raise IrisRunStateError(f"示例没有完成: {result.error}", run_id=result.run.run_id)
        calls = runner.list_tool_calls(result.run.run_id)
        errors = [call.tool_call_id for call in calls if call.result and call.result.is_error]
        if errors != ["python-broken"] or not provider.error_feedback:
            raise IrisToolExecutionError("示例未完成预期的错误反馈与修正流程")
        artifact = calls[-1].result.artifact
        if artifact is None:
            raise IrisToolExecutionError("示例没有返回已发布的报告副本")
        return {
            "workspace": str(workspace),
            "tool_names": [call.tool_name for call in calls],
            "confirmed_tools": confirmed,
            "output": json.loads(artifact.path.read_text(encoding="utf-8")),
            "error_feedback": provider.error_feedback,
            "artifact": artifact.model_dump(mode="json"),
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
