"""DeepSeek session live 验证场景。"""

from __future__ import annotations

from pathlib import Path

from iris.runtime import RuntimeFactory
from iris.runtime.models import BoundedLoopOptions, RuntimeOptions

from .fixtures import prepare_read_agent
from .models import ScenarioReport
from .providers import recording_provider
from .reporting import scenario_report
from .utils import _retry_assertion, _runtime_error_code, _runtime_error_message


async def run_sqlite_session_live(work_dir: Path, retries: int) -> ScenarioReport:
    """验证 SQLite session 持久化 messages、run metadata 和 tool events。"""
    agent_path, token = prepare_read_agent(work_dir, session_backend="sqlite")
    session_id = "sqlite-session-live"

    async def attempt() -> ScenarioReport:
        provider = recording_provider()
        runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
        result = await runtime.run_loop(
            (
                "必须先调用 read_file 读取 verification.txt。读取后只输出一行："
                f"SQLITE_SESSION_OK: {token}"
            ),
            options=RuntimeOptions(
                session_id=session_id,
                loop=BoundedLoopOptions(max_steps=4),
                metadata={"scenario": "sqlite_session_live"},
            ),
        )
        messages = runtime.session_store.load_messages(session_id)
        metadata = runtime.session_store.load_run_metadata(session_id)
        tool_events = runtime.session_store.load_tool_events(session_id)
        final_text = (
            result.assistant_message.text.strip() if result.assistant_message else ""
        )
        db_path = work_dir / "session.db"
        ok = (
            result.status.value == "ok"
            and db_path.exists()
            and len(messages) >= 3
            and metadata.get("latest_run", {}).get("status") == "ok"
            and len(tool_events) >= 1
            and token in final_text
        )
        return scenario_report(
            name="sqlite_session_live",
            ok=ok,
            status=result.status.value,
            api_calls=provider.api_call_count,
            steps=result.steps,
            expected="SQLite messages/latest_run/tool_events 均写入",
            actual=f"messages={len(messages)}; tool_events={len(tool_events)}",
            evidence={
                "db_path": str(db_path),
                "latest_run": metadata.get("latest_run", {}),
                "request_snapshots": provider.request_snapshots(),
            },
            error_code="" if ok else _runtime_error_code(result),
            error_message=(
                ""
                if ok
                else _runtime_error_message(
                    result,
                    "SQLite session 验证失败",
                )
            ),
        )

    return await _retry_assertion(attempt, retries)
