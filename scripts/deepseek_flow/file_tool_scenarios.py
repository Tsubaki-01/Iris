"""DeepSeek 内置文件工具 live 验证场景。"""

from __future__ import annotations

from pathlib import Path

from iris.runtime import RuntimeFactory
from iris.runtime.models import BoundedLoopOptions, RuntimeOptions

from .fixtures import prepare_all_file_tools_agent, prepare_file_not_read_recovery_agent
from .models import ScenarioReport
from .providers import recording_provider
from .reporting import scenario_report
from .utils import _retry_assertion, _tool_choice


async def run_builtin_file_tools_live(work_dir: Path, retries: int) -> ScenarioReport:
    """验证真实 API 分别调用内置文件工具。"""

    async def attempt() -> ScenarioReport:
        agent_path = prepare_all_file_tools_agent(work_dir)
        provider = recording_provider()
        runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
        session_id = "builtin-file-tools-live"
        calls = [
            (
                "list_files",
                "调用 list_files，参数 path='.'，max_results=20。",
            ),
            (
                "read_file",
                "调用 read_file 读取 notes.txt。",
            ),
            (
                "grep_search",
                "调用 grep_search，参数 pattern='ALPHA_PATTERN_0708'，path='notes.txt'。",
            ),
            (
                "write_file",
                (
                    "调用 write_file，参数 file_path='generated.txt'，"
                    "content='LIVE_WRITE_TOKEN_0708'。"
                ),
            ),
            (
                "edit_file",
                (
                    "调用 edit_file，参数 file_path='notes.txt'，"
                    "old_string='old-value'，new_string='new-value'。"
                ),
            ),
        ]
        tool_codes: list[str] = []
        loop_statuses: list[str] = []
        loop_error_codes: list[str] = []
        for tool_name, prompt in calls:
            result = await runtime.run_loop(
                prompt,
                options=RuntimeOptions(
                    session_id=session_id,
                    loop=BoundedLoopOptions(max_steps=1),
                    request_options={"tool_choice": _tool_choice(tool_name)},
                    metadata={
                        "scenario": "builtin_file_tools_live",
                        "tool_name": tool_name,
                    },
                ),
            )
            loop_statuses.append(result.status.value)
            loop_error_codes.append(result.error.code if result.error else "")
            if not result.tool_results:
                tool_codes.append("NO_TOOL_RESULT")
                continue
            tool_result = result.tool_results[0]
            tool_codes.append(
                "ok"
                if not tool_result.is_error
                else tool_result.error.code if tool_result.error else "ERROR"
            )

        notes_text = (work_dir / "workspace" / "notes.txt").read_text(encoding="utf-8")
        generated_path = work_dir / "workspace" / "generated.txt"
        generated_text = (
            generated_path.read_text(encoding="utf-8")
            if generated_path.exists()
            else ""
        )
        ok = (
            tool_codes == ["ok", "ok", "ok", "ok", "ok"]
            and "new-value" in notes_text
            and generated_text == "LIVE_WRITE_TOKEN_0708"
        )
        return scenario_report(
            name="builtin_file_tools_live",
            ok=ok,
            status="ok" if ok else "assertion_failed",
            api_calls=provider.api_call_count,
            steps=len(calls),
            expected="list/read/grep/write/edit 全部成功",
            actual=", ".join(tool_codes),
            evidence={
                "notes_text": notes_text,
                "generated_text": generated_text,
                "loop_statuses": loop_statuses,
                "loop_error_codes": loop_error_codes,
                "request_snapshots": provider.request_snapshots(),
            },
            error_code="" if ok else "ASSERTION_FAILED",
            error_message="" if ok else "内置文件工具 live 调用未全部成功",
        )

    return await _retry_assertion(attempt, retries)


async def run_file_not_read_recovery_live(
    work_dir: Path,
    retries: int,
) -> ScenarioReport:
    """验证模型收到 FILE_NOT_READ 后会先读文件再重试写入。"""

    async def attempt() -> ScenarioReport:
        agent_path = prepare_file_not_read_recovery_agent(work_dir)
        provider = recording_provider()
        runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
        target_path = work_dir / "workspace" / "target.txt"
        target_token = "RECOVERED_WRITE_TOKEN_0708"
        result = await runtime.run_loop(
            (
                "这是 FILE_NOT_READ 恢复能力验证。第一步必须先调用 write_file "
                f"覆盖 target.txt，content='{target_token}'。"
                "如果工具返回 FILE_NOT_READ，下一步必须调用 read_file 读取 target.txt，"
                "然后再次调用 write_file 写入相同 content。不要直接回答。"
            ),
            options=RuntimeOptions(
                session_id="file-not-read-recovery-live",
                loop=BoundedLoopOptions(max_steps=3),
                request_options={"tool_choice": "auto"},
                metadata={"scenario": "file_not_read_recovery_live"},
            ),
        )
        tool_sequence = [tool_result.tool_name for tool_result in result.tool_results]
        tool_codes = [
            (
                "ok"
                if not tool_result.is_error
                else tool_result.error.code if tool_result.error else "ERROR"
            )
            for tool_result in result.tool_results
        ]
        target_text = target_path.read_text(encoding="utf-8")
        expected_sequence = ["write_file", "read_file", "write_file"]
        expected_codes = ["FILE_NOT_READ", "ok", "ok"]
        ok = (
            tool_sequence == expected_sequence
            and tool_codes == expected_codes
            and target_text == target_token
        )
        actual = " -> ".join(
            f"{tool_name}:{tool_code}"
            for tool_name, tool_code in zip(tool_sequence, tool_codes, strict=False)
        )
        return scenario_report(
            name="file_not_read_recovery_live",
            ok=ok,
            status="ok" if ok else "assertion_failed",
            api_calls=provider.api_call_count,
            steps=result.steps,
            expected="write_file:FILE_NOT_READ -> read_file:ok -> write_file:ok",
            actual=actual,
            evidence={
                "agent_path": str(agent_path),
                "target_file": str(target_path.resolve()),
                "target_text": target_text,
                "tool_sequence": tool_sequence,
                "tool_codes": tool_codes,
                "loop_status": result.status.value,
                "loop_error_code": result.error.code if result.error else "",
                "request_snapshots": provider.request_snapshots(),
            },
            error_code="" if ok else "ASSERTION_FAILED",
            error_message=(
                "" if ok else "模型未在 FILE_NOT_READ 后按 read_file -> write_file 恢复"
            ),
        )

    return await _retry_assertion(attempt, retries)
