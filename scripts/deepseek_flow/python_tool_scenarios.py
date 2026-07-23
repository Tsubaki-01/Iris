"""DeepSeek Python 自定义工具 live 验证场景。"""

from __future__ import annotations

from pathlib import Path

from iris.runtime import RuntimeFactory
from iris.runtime.models import BoundedLoopOptions, RuntimeOptions

from .fixtures import _write_agent_yaml, _write_python_tool_module
from .models import ScenarioReport
from .providers import recording_provider
from .reporting import scenario_report
from .utils import _retry_assertion, _runtime_error_code, _runtime_error_message


async def run_python_tool_live(work_dir: Path, retries: int) -> ScenarioReport:
    """验证真实 API 调用 YAML 注册的 Python 工具。"""
    _write_python_tool_module(work_dir)
    agent_path = _write_agent_yaml(
        work_dir,
        """
name: python-tool-live
model:
  provider: deepseek
  name: deepseek-chat
  temperature: 0
  max_tokens: 160
system: |
  你是 Python tool live 验证助手。需要查询笔记时必须调用 search_notes。
tools:
  python:
    functions:
      - deepseek_live_tools:search_notes
permissions:
  workspace: workspace
  writes: deny
session:
  backend: none
""",
    )

    async def attempt() -> ScenarioReport:
        provider = recording_provider()
        runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
        result = await runtime.run_loop(
            (
                "调用 search_notes 查询 deepseek，然后最终只回答 "
                "PYTHON_TOOL_OK: PYTHON_TOOL_TOKEN_0708"
            ),
            options=RuntimeOptions(
                session_id="python-tool-live",
                loop=BoundedLoopOptions(max_steps=3),
                request_options={"tool_choice": "auto"},
                metadata={"scenario": "python_tool_live"},
            ),
        )
        final_text = result.assistant_message.text.strip() if result.assistant_message else ""
        tool_contents = [tool_result.model_content for tool_result in result.tool_results]
        ok = (
            result.status.value == "ok"
            and any("PYTHON_TOOL_TOKEN_0708" in content for content in tool_contents)
            and "PYTHON_TOOL_OK" in final_text
            and "PYTHON_TOOL_TOKEN_0708" in final_text
        )
        return scenario_report(
            name="python_tool_live",
            ok=ok,
            status=result.status.value,
            api_calls=provider.api_call_count,
            steps=result.steps,
            expected="search_notes 工具结果回灌并生成最终回答",
            actual=final_text,
            evidence={
                "tool_contents": tool_contents,
                "request_snapshots": provider.request_snapshots(),
            },
            error_code="" if ok else _runtime_error_code(result),
            error_message=(
                ""
                if ok
                else _runtime_error_message(
                    result,
                    "Python 工具 live 验证失败",
                )
            ),
        )

    return await _retry_assertion(attempt, retries)
