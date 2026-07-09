"""DeepSeek runtime live 验证场景。"""

from __future__ import annotations

from pathlib import Path

from iris.runtime import RuntimeFactory
from iris.runtime.models import BoundedLoopOptions, RuntimeOptions

from .constants import RUNTIME_OK_PREFIX
from .fixtures import _write_agent_yaml, prepare_read_agent
from .models import ScenarioReport
from .providers import recording_provider
from .reporting import scenario_report
from .utils import (
    _request_has_tool_result,
    _retry_assertion,
    _runtime_error_code,
    _runtime_error_message,
    _runtime_final_ok,
    _tool_choice,
)


async def run_runtime_read_loop_live(work_dir: Path, retries: int) -> ScenarioReport:
    """验证真实 run_loop 的 file.read 工具调用和最终回答。"""
    agent_path, token = prepare_read_agent(work_dir, session_backend="none")

    async def attempt() -> ScenarioReport:
        provider = recording_provider()
        runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
        result = await runtime.run_loop(
            (
                "必须先调用 read_file 读取 verification.txt。读取后只输出一行："
                f"{RUNTIME_OK_PREFIX} <文件中的验证码>"
            ),
            options=RuntimeOptions(
                session_id="runtime-read-loop-live",
                loop=BoundedLoopOptions(max_steps=4),
                metadata={"scenario": "runtime_read_loop_live"},
            ),
        )
        final_text = (
            result.assistant_message.text.strip() if result.assistant_message else ""
        )
        ok = (
            result.status.value == "ok"
            and len(result.tool_results) > 0
            and _runtime_final_ok(final_text, token)
        )
        return scenario_report(
            name="runtime_read_loop_live",
            ok=ok,
            status=result.status.value,
            api_calls=provider.api_call_count,
            steps=result.steps,
            expected=f"{RUNTIME_OK_PREFIX} {token}",
            actual=final_text,
            evidence={
                "tool_result_count": len(result.tool_results),
                "request_has_tool_result": any(
                    _request_has_tool_result(request) for request in provider.requests
                ),
                "request_snapshots": provider.request_snapshots(),
            },
            error_code="" if ok else _runtime_error_code(result),
            error_message=(
                "" if ok else _runtime_error_message(result, "未得到严格最终回答")
            ),
        )

    return await _retry_assertion(attempt, retries)


async def run_run_turn_live(work_dir: Path, retries: int) -> ScenarioReport:
    """验证真实 run_turn 只发起一次 provider 调用且执行一次工具桥接。"""
    agent_path, _ = prepare_read_agent(work_dir, session_backend="none")

    async def attempt() -> ScenarioReport:
        provider = recording_provider()
        runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
        result = await runtime.run_turn(
            "调用 read_file 读取 verification.txt。不要直接回答，必须调用工具。",
            options=RuntimeOptions(
                session_id="run-turn-live",
                request_options={"tool_choice": _tool_choice("read_file")},
                metadata={"scenario": "run_turn_live"},
            ),
        )
        tool_events = runtime.session_store.load_tool_events("run-turn-live")
        ok = (
            result.status.value == "ok"
            and provider.api_call_count == 1
            and len(result.tool_results) == 1
            and len(tool_events) == 1
        )
        return scenario_report(
            name="run_turn_live",
            ok=ok,
            status=result.status.value,
            api_calls=provider.api_call_count,
            steps=result.steps,
            expected="一次 provider 调用 + 一次 read_file 工具结果",
            actual=f"api_calls={provider.api_call_count}; tool_results={len(result.tool_results)}",
            evidence={
                "tool_events": tool_events,
                "request_snapshots": provider.request_snapshots(),
            },
            error_code="" if ok else _runtime_error_code(result),
            error_message=(
                ""
                if ok
                else _runtime_error_message(
                    result,
                    "run_turn 语义不符合预期",
                )
            ),
        )

    return await _retry_assertion(attempt, retries)


async def run_context_yaml_live(work_dir: Path, retries: int) -> ScenarioReport:
    """验证结构化 context.yaml 会进入真实 API 请求。"""
    del retries
    context_path = work_dir / "context.yaml"
    context_path.write_text(
        """
system:
  slots:
    - name: instructions
      content: 你是 Iris context live 验证助手。
memory:
  slots:
    - name: memory
      content: CONTEXT_MEMORY_TOKEN_0708
before_current_input:
  slots:
    - name: environment_state
      content: CONTEXT_BEFORE_TOKEN_0708
""".strip(),
        encoding="utf-8",
    )
    agent_path = _write_agent_yaml(
        work_dir,
        """
name: context-yaml-live
model:
  provider: deepseek
  name: deepseek-chat
  temperature: 0
  max_tokens: 64
context:
  path: context.yaml
session:
  backend: none
""",
    )
    provider = recording_provider()
    runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
    result = await runtime.run_loop(
        "只回答 CONTEXT_YAML_OK。",
        options=RuntimeOptions(session_id="context-yaml-live"),
    )
    request = provider.requests[0]
    texts = [message.text for message in request.messages]
    roles = [message.role.value for message in request.messages]
    ok = (
        result.status.value == "ok"
        and roles[:4] == ["system", "user", "user", "user"]
        and "CONTEXT_MEMORY_TOKEN_0708" in texts[1]
        and "CONTEXT_BEFORE_TOKEN_0708" in texts[2]
        and texts[3] == "只回答 CONTEXT_YAML_OK。"
    )
    return scenario_report(
        name="context_yaml_live",
        ok=ok,
        status=result.status.value,
        api_calls=provider.api_call_count,
        steps=result.steps,
        expected="system -> memory -> before_current_input -> current input",
        actual=" -> ".join(roles),
        evidence={
            "message_texts": texts,
            "request_snapshots": provider.request_snapshots(),
        },
        error_code="" if ok else _runtime_error_code(result),
        error_message=(
            ""
            if ok
            else _runtime_error_message(
                result,
                "context 消息顺序不符合预期",
            )
        ),
    )
