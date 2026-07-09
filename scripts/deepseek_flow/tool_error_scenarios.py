"""DeepSeek 工具错误路径 live 验证场景。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iris.runtime import RuntimeFactory
from iris.runtime.models import BoundedLoopOptions, RuntimeOptions, ToolErrorPolicy
from iris.session import InMemorySessionStore

from .constants import DEFAULT_MODEL
from .fixtures import _write_agent_yaml, prepare_read_agent
from .models import ScenarioReport
from .providers import recording_provider
from .reporting import scenario_report
from .utils import _first_tool_error_code, _tool_choice


async def run_tool_errors_live(work_dir: Path, retries: int) -> ScenarioReport:
    """验证真实 API 触发工具错误路径。"""
    del retries
    permission = await _permission_error_probe(work_dir / "permission")
    file_not_read = await _file_not_read_probe(work_dir / "file-not-read")
    max_steps = await _max_steps_probe(work_dir / "max-steps")
    tool_not_allowed = await _tool_not_allowed_probe(work_dir / "tool-not-allowed")
    probes = {
        "permission_error": permission,
        "file_not_read": file_not_read,
        "max_steps": max_steps,
        "tool_not_allowed": tool_not_allowed,
    }
    ok = all(probe["ok"] for probe in probes.values())
    return scenario_report(
        name="tool_errors_live",
        ok=ok,
        status="ok" if ok else "assertion_failed",
        api_calls=sum(probe["api_calls"] for probe in probes.values()),
        steps=sum(probe["steps"] for probe in probes.values()),
        expected="PERMISSION_ERROR/FILE_NOT_READ/MAX_STEPS_REACHED/TOOL_NOT_ALLOWED",
        actual=", ".join(f"{name}:{probe['actual']}" for name, probe in probes.items()),
        evidence=probes,
        error_code="" if ok else "ASSERTION_FAILED",
        error_message="" if ok else "工具错误 live 验证失败",
    )


async def run_permission_path_escape_live(
    work_dir: Path,
    retries: int,
) -> ScenarioReport:
    """验证 agent.yaml permissions.workspace 会拒绝父目录路径逃逸。"""
    del retries
    work_dir.mkdir(parents=True, exist_ok=True)
    workspace = work_dir / "workspace"
    workspace.mkdir(exist_ok=True)
    outside_file = work_dir / "outside-secret.txt"
    outside_file.write_text("PATH_ESCAPE_SECRET_SHOULD_NOT_BE_READ", encoding="utf-8")
    agent_path = _write_agent_yaml(
        work_dir,
        f"""
name: permission-path-escape-live
model:
  provider: deepseek
  name: {DEFAULT_MODEL}
  temperature: 0
  max_tokens: 120
system: 必须调用用户指定的文件工具。
tools:
  builtin:
    - file.read
permissions:
  workspace: workspace
  writes: deny
session:
  backend: none
""",
    )
    provider = recording_provider()
    runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
    attempted_path = "../outside-secret.txt"
    result = await runtime.run_loop(
        f"调用 read_file，参数 file_path='{attempted_path}'。",
        options=RuntimeOptions(
            session_id="permission-path-escape-live",
            loop=BoundedLoopOptions(
                max_steps=1,
                tool_error_policy=ToolErrorPolicy.STOP,
            ),
            request_options={"tool_choice": _tool_choice("read_file")},
            metadata={"scenario": "permission_path_escape_live"},
        ),
    )
    code = _first_tool_error_code(result.tool_results)
    model_contents = [tool_result.model_content for tool_result in result.tool_results]
    boundary_code = (
        "PATH_OUTSIDE_WORKSPACE"
        if any("PATH_OUTSIDE_WORKSPACE" in content for content in model_contents)
        else code
    )
    ok = code == "VALIDATION_ERROR" and boundary_code == "PATH_OUTSIDE_WORKSPACE"
    return scenario_report(
        name="permission_path_escape_live",
        ok=ok,
        status="ok" if ok else "assertion_failed",
        api_calls=provider.api_call_count,
        steps=result.steps,
        expected="VALIDATION_ERROR containing PATH_OUTSIDE_WORKSPACE",
        actual=boundary_code,
        evidence={
            "agent_path": str(agent_path),
            "workspace_root": str(workspace.resolve()),
            "outside_file": str(outside_file.resolve()),
            "outside_file_exists": outside_file.exists(),
            "outside_file_readable_by_tool": any(
                "PATH_ESCAPE_SECRET_SHOULD_NOT_BE_READ" in content
                for content in model_contents
            ),
            "attempted_path": attempted_path,
            "executor_error_code": code,
            "boundary_error_code": boundary_code,
            "model_contents": model_contents,
            "request_snapshots": provider.request_snapshots(),
        },
        error_code="" if ok else "ASSERTION_FAILED",
        error_message="" if ok else "路径逃逸未被 workspace 策略拒绝",
    )


async def _permission_error_probe(work_dir: Path) -> dict[str, Any]:
    """触发写权限拒绝。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    workspace = work_dir / "workspace"
    workspace.mkdir(exist_ok=True)
    agent_path = _write_agent_yaml(
        work_dir,
        f"""
name: permission-error-live
model:
  provider: deepseek
  name: {DEFAULT_MODEL}
  temperature: 0
  max_tokens: 120
system: 必须调用用户指定的文件工具。
tools:
  builtin:
    - file.write
permissions:
  workspace: workspace
  writes: deny
session:
  backend: none
""",
    )
    provider = recording_provider()
    runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
    result = await runtime.run_loop(
        "调用 write_file，参数 file_path='denied.txt'，content='denied'。",
        options=RuntimeOptions(
            session_id="permission-error-live",
            loop=BoundedLoopOptions(
                max_steps=1,
                tool_error_policy=ToolErrorPolicy.STOP,
            ),
            request_options={"tool_choice": _tool_choice("write_file")},
        ),
    )
    code = _first_tool_error_code(result.tool_results)
    return {
        "ok": code == "PERMISSION_ERROR",
        "actual": code,
        "api_calls": provider.api_call_count,
        "steps": result.steps,
        "status": result.status.value,
        "request_snapshots": provider.request_snapshots(),
    }


async def _file_not_read_probe(work_dir: Path) -> dict[str, Any]:
    """触发写入已有文件前未读取错误。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    workspace = work_dir / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "existing.txt").write_text("old", encoding="utf-8")
    agent_path = _write_agent_yaml(
        work_dir,
        f"""
name: file-not-read-live
model:
  provider: deepseek
  name: {DEFAULT_MODEL}
  temperature: 0
  max_tokens: 120
system: 必须调用用户指定的文件工具。
tools:
  builtin:
    - file.write
permissions:
  workspace: workspace
  writes: allow
session:
  backend: none
""",
    )
    provider = recording_provider()
    runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
    result = await runtime.run_loop(
        "调用 write_file，参数 file_path='existing.txt'，content='new'。",
        options=RuntimeOptions(
            session_id="file-not-read-live",
            loop=BoundedLoopOptions(
                max_steps=1,
                tool_error_policy=ToolErrorPolicy.STOP,
            ),
            request_options={"tool_choice": _tool_choice("write_file")},
        ),
    )
    code = _first_tool_error_code(result.tool_results)
    return {
        "ok": code == "FILE_NOT_READ",
        "actual": code,
        "api_calls": provider.api_call_count,
        "steps": result.steps,
        "status": result.status.value,
        "request_snapshots": provider.request_snapshots(),
    }


async def _max_steps_probe(work_dir: Path) -> dict[str, Any]:
    """触发 bounded loop 步数上限。"""
    agent_path, _ = prepare_read_agent(work_dir, session_backend="none")
    provider = recording_provider()
    runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
    result = await runtime.run_loop(
        "调用 read_file 读取 verification.txt。",
        options=RuntimeOptions(
            session_id="max-steps-live",
            loop=BoundedLoopOptions(max_steps=1),
            request_options={"tool_choice": _tool_choice("read_file")},
        ),
    )
    code = result.error.code if result.error else ""
    return {
        "ok": code == "MAX_STEPS_REACHED",
        "actual": code,
        "api_calls": provider.api_call_count,
        "steps": result.steps,
        "status": result.status.value,
        "request_snapshots": provider.request_snapshots(),
    }


async def _tool_not_allowed_probe(work_dir: Path) -> dict[str, Any]:
    """用真实 API 生成合法工具调用，再验证禁用工具桥接结果。"""
    agent_path, _ = prepare_read_agent(work_dir, session_backend="none")
    provider = recording_provider()
    store = InMemorySessionStore()
    runtime = RuntimeFactory.from_config_path(
        agent_path,
        provider=provider,
        session_store=store,
    )
    result = await runtime.run_loop(
        "调用 read_file 读取 verification.txt。",
        options=RuntimeOptions(
            session_id="tool-not-allowed-source",
            loop=BoundedLoopOptions(max_steps=1),
            request_options={"tool_choice": _tool_choice("read_file")},
        ),
    )
    if result.assistant_message is None:
        return {
            "ok": False,
            "actual": "NO_ASSISTANT",
            "api_calls": provider.api_call_count,
            "steps": result.steps,
            "status": result.status.value,
            "request_snapshots": provider.request_snapshots(),
        }
    bridge_result = await runtime.tool_bridge.execute_once(
        assistant_message=result.assistant_message,
        session_id="tool-not-allowed-local",
        run_id="tool-not-allowed-run",
        step_index=0,
        agent_id="tool-not-allowed-agent",
        workspace_root=work_dir / "workspace",
        permission_mode="deny",
        session_store=store,
        metadata={"scenario": "tool_errors_live"},
        tools_enabled=False,
    )
    code = _first_tool_error_code(bridge_result.results)
    return {
        "ok": code == "TOOL_NOT_ALLOWED",
        "actual": code,
        "api_calls": provider.api_call_count,
        "steps": result.steps,
        "status": result.status.value,
        "request_snapshots": provider.request_snapshots(),
    }
