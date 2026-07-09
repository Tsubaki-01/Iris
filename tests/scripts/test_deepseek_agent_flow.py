from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import pytest

import iris
from iris.message import LLMRequest, LLMResponse, Msg, TextBlock, ToolUseBlock

LIVE_DEEPSEEK_GATE_SCENARIOS = (
    "provider_smoke_live",
    "runtime_read_loop_live",
    "builtin_file_tools_live",
    "file_not_read_recovery_live",
)


@pytest.fixture(autouse=True)
def reset_config_state(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """隔离脚本测试中的全局配置状态。"""
    if request.node.get_closest_marker("live_deepseek") is None:
        for name in (
            "IRIS_API_KEY",
            "IRIS_PROVIDER_API_KEYS__DEEPSEEK",
        ):
            monkeypatch.delenv(name, raising=False)
    iris.reset()
    yield
    iris.reset()


class DelegateProvider:
    """测试用真实 provider 替身，只用于验证 recording wrapper 行为。"""

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """记录调用并返回固定响应。"""
        self.calls.append(request)
        return LLMResponse(
            provider="delegate",
            id="response-1",
            model=request.model,
            content=[TextBlock(text="ok")],
            finish_reason="stop",
        )


def _load_demo_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "deepseek_agent_flow.py"
    )
    spec = importlib.util.spec_from_file_location("deepseek_agent_flow", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_flow_module(name: str) -> ModuleType:
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module(name)


def test_init_local_config_loads_api_key_without_overriding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_flow_module("deepseek_flow.config")
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "IRIS_API_KEY=local-key\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("IRIS_PROVIDER_API_KEYS__DEEPSEEK", raising=False)
    monkeypatch.setenv("IRIS_API_KEY", "existing-key")

    initialized = module.init_local_config(tmp_path)

    assert initialized is True
    assert module.resolve_api_key() == "existing-key"
    assert iris.get_config().api_key == "existing-key"


def test_init_local_config_loads_provider_key_from_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_flow_module("deepseek_flow.config")
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "IRIS_PROVIDER_API_KEYS__DEEPSEEK=deepseek-key\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("IRIS_API_KEY", raising=False)
    monkeypatch.delenv("IRIS_PROVIDER_API_KEYS__DEEPSEEK", raising=False)

    initialized = module.init_local_config(tmp_path)

    assert initialized is True
    assert module.resolve_api_key() == "deepseek-key"
    assert iris.get_config().provider_api_keys["deepseek"] == "deepseek-key"


def test_parse_args_defaults_to_all_live_scenarios() -> None:
    module = _load_flow_module("deepseek_flow.cli")

    args = module.parse_args([])

    assert args.scenario == "all"
    assert args.retries == 2
    assert args.work_dir is None


def test_scenario_names_include_only_live_scenarios() -> None:
    module = _load_flow_module("deepseek_flow.constants")

    assert set(module.SCENARIO_NAMES) == {
        "provider_smoke_live",
        "runtime_read_loop_live",
        "run_turn_live",
        "context_yaml_live",
        "memory_results_live",
        "memory_query_live",
        "sqlite_session_live",
        "builtin_file_tools_live",
        "file_not_read_recovery_live",
        "python_tool_live",
        "permission_path_escape_live",
        "tool_errors_live",
    }


def test_scenario_catalog_describes_each_live_scenario() -> None:
    constants = _load_flow_module("deepseek_flow.constants")
    catalog = _load_flow_module("deepseek_flow.catalog")

    assert set(catalog.SCENARIO_CATALOG) == set(constants.SCENARIO_NAMES)
    assert catalog.SCENARIO_CATALOG["runtime_read_loop_live"] == {
        "module": "runtime",
        "runtime_api": "run_loop",
        "uses_deepseek": True,
        "description": "验证 file.read 工具调用、tool result 回灌和最终回答。",
    }
    assert catalog.SCENARIO_CATALOG["run_turn_live"]["runtime_api"] == "run_turn"
    assert catalog.SCENARIO_CATALOG["provider_smoke_live"]["module"] == "providers"
    assert (
        catalog.SCENARIO_CATALOG["permission_path_escape_live"]["module"]
        == "tools.permissions"
    )
    assert (
        catalog.SCENARIO_CATALOG["file_not_read_recovery_live"]["description"]
        == "验证模型收到 FILE_NOT_READ 后会先读文件再重试写入。"
    )


def test_only_run_turn_scenario_uses_run_turn_api() -> None:
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts" / "deepseek_flow"
    run_turn_usages = {
        path.name: [
            line_number
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            )
            if ".run_turn(" in line
        ]
        for path in scripts_dir.glob("*.py")
    }

    non_empty_usages = {
        file_name: lines for file_name, lines in run_turn_usages.items() if lines
    }
    assert set(non_empty_usages) == {"runtime_scenarios.py"}
    assert len(non_empty_usages["runtime_scenarios.py"]) == 1


def test_strict_provider_smoke_requires_exact_token() -> None:
    module = _load_flow_module("deepseek_flow.utils")

    assert module._provider_smoke_ok("IRIS_PROVIDER_OK") is True
    assert module._provider_smoke_ok(" IRIS_PROVIDER_OK ") is True
    assert module._provider_smoke_ok("IRIS_PROVIDER_OK.") is False
    assert module._provider_smoke_ok("ok") is False


def test_strict_runtime_final_answer_requires_prefix_and_token() -> None:
    module = _load_flow_module("deepseek_flow.utils")
    token = "deepseek-flow-token-test"

    assert module._runtime_final_ok(f"IRIS_RUNTIME_TOOL_OK: {token}", token) is True
    assert (
        module._runtime_final_ok(f"prefix IRIS_RUNTIME_TOOL_OK: {token}", token)
        is False
    )
    assert (
        module._runtime_final_ok(f"IRIS_RUNTIME_TOOL_OK: {token}\nextra", token)
        is False
    )
    assert module._runtime_final_ok("IRIS_RUNTIME_TOOL_OK: missing", token) is False


def test_safe_error_message_redacts_current_api_key() -> None:
    module = _load_flow_module("deepseek_flow.config")
    module.resolve_api_key = lambda: "sk-live-secret"

    message = module._safe_error_message(RuntimeError("failed with sk-live-secret"))

    assert message == "failed with sk-...cret"
    assert "sk-live-secret" not in message


def test_recording_runtime_provider_delegates_and_records_requests() -> None:
    module = _load_flow_module("deepseek_flow.providers")
    delegate = DelegateProvider()
    provider = module.RecordingRuntimeProvider(delegate)
    request = LLMRequest(model="deepseek-chat", messages=[Msg.user("你好")])

    response = asyncio.run(provider.complete(request))

    assert response.to_msg().text == "ok"
    assert delegate.calls == [request]
    assert provider.requests == [request]
    assert provider.api_call_count == 1


def test_recording_runtime_provider_exposes_request_snapshots() -> None:
    module = _load_flow_module("deepseek_flow.providers")
    delegate = DelegateProvider()
    provider = module.RecordingRuntimeProvider(delegate)
    request = LLMRequest(
        model="deepseek-chat",
        messages=[
            Msg.system("系统规则"),
            Msg.user("读取文件"),
            Msg.tool_result(
                tool_use_id="call-read",
                content="文件内容",
                name="read_file",
            ),
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "read_file"},
            }
        ],
        tool_choice={"type": "function", "function": {"name": "read_file"}},
        temperature=0,
        max_tokens=128,
    )

    asyncio.run(provider.complete(request))

    assert provider.request_snapshots() == [
        {
            "index": 1,
            "model": "deepseek-chat",
            "message_count": 3,
            "roles": ["system", "user", "user"],
            "tool_schema_names": ["read_file"],
            "tool_choice": {
                "type": "function",
                "function": {"name": "read_file"},
            },
            "temperature": 0,
            "max_tokens": 128,
            "has_tool_result": True,
            "messages": [
                {
                    "role": "system",
                    "text_preview": "系统规则",
                    "tool_call_names": [],
                    "tool_result_names": [],
                    "has_tool_result": False,
                },
                {
                    "role": "user",
                    "text_preview": "读取文件",
                    "tool_call_names": [],
                    "tool_result_names": [],
                    "has_tool_result": False,
                },
                {
                    "role": "user",
                    "text_preview": "文件内容",
                    "tool_call_names": [],
                    "tool_result_names": ["read_file"],
                    "has_tool_result": True,
                },
            ],
        }
    ]


def test_provider_smoke_reports_attempted_api_call_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_flow_module("deepseek_flow.providers")

    class FailingClient:
        async def complete(self, _request: LLMRequest) -> LLMResponse:
            raise RuntimeError("network failed")

    monkeypatch.setattr(
        module,
        "create_provider_client",
        lambda *args, **kwargs: FailingClient(),
    )
    monkeypatch.setattr(module, "_safe_error_message", lambda exc: "network failed")

    report = asyncio.run(module.run_provider_smoke_live(tmp_path, retries=0))

    assert report["ok"] is False
    assert report["status"] == "error"
    assert report["api_calls"] == 1
    assert report["expected"] == "IRIS_PROVIDER_OK"
    assert report["actual"] == "RuntimeError"
    assert report["error_code"] == "RuntimeError"
    assert report["error_message"] == "network failed"
    assert report["evidence"]["provider_route"] == "deepseek/deepseek-chat"


def test_permission_path_escape_live_reports_workspace_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_flow_module("deepseek_flow.tool_error_scenarios")
    providers = _load_flow_module("deepseek_flow.providers")

    class PathEscapeProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.calls += 1
            return LLMResponse(
                provider="fake",
                id="path-escape",
                model=request.model,
                content=[
                    ToolUseBlock(
                        id="call-read-escape",
                        name="read_file",
                        input={"file_path": "../outside-secret.txt"},
                    )
                ],
                finish_reason="tool_calls",
            )

    monkeypatch.setattr(
        providers,
        "create_provider_client",
        lambda *args, **kwargs: PathEscapeProvider(),
    )

    report = asyncio.run(module.run_permission_path_escape_live(tmp_path, retries=0))

    assert report["ok"] is True
    assert report["actual"] == "PATH_OUTSIDE_WORKSPACE"
    assert report["api_calls"] == 1
    assert report["steps"] == 1
    assert report["evidence"]["outside_file_exists"] is True
    assert report["evidence"]["outside_file_readable_by_tool"] is False
    assert report["evidence"]["workspace_root"].endswith("workspace")
    assert report["evidence"]["attempted_path"] == "../outside-secret.txt"


def test_builtin_file_tools_live_resets_stale_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_flow_module("deepseek_flow.file_tool_scenarios")
    providers = _load_flow_module("deepseek_flow.providers")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "generated.txt").write_text("stale", encoding="utf-8")

    class FileToolProvider:
        async def complete(self, request: LLMRequest) -> LLMResponse:
            tool_choice = request.tool_choice
            assert isinstance(tool_choice, dict)
            function = tool_choice["function"]
            assert isinstance(function, dict)
            tool_name = function["name"]
            inputs = {
                "list_files": {"path": ".", "max_results": 20},
                "read_file": {"file_path": "notes.txt"},
                "grep_search": {
                    "pattern": "ALPHA_PATTERN_0708",
                    "path": "notes.txt",
                },
                "write_file": {
                    "file_path": "generated.txt",
                    "content": "LIVE_WRITE_TOKEN_0708",
                },
                "edit_file": {
                    "file_path": "notes.txt",
                    "old_string": "old-value",
                    "new_string": "new-value",
                },
            }
            return LLMResponse(
                provider="fake",
                id=f"file-tool-{tool_name}",
                model=request.model,
                content=[
                    ToolUseBlock(
                        id=f"call-{tool_name}",
                        name=str(tool_name),
                        input=inputs[str(tool_name)],
                    )
                ],
                finish_reason="tool_calls",
            )

    monkeypatch.setattr(
        providers,
        "create_provider_client",
        lambda *args, **kwargs: FileToolProvider(),
    )

    report = asyncio.run(module.run_builtin_file_tools_live(tmp_path, retries=0))

    assert report["ok"] is True
    assert report["actual"] == "ok, ok, ok, ok, ok"
    assert report["evidence"]["generated_text"] == "LIVE_WRITE_TOKEN_0708"
    assert "new-value" in report["evidence"]["notes_text"]


def test_file_not_read_recovery_live_reports_read_then_write_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_flow_module("deepseek_flow.file_tool_scenarios")
    providers = _load_flow_module("deepseek_flow.providers")

    class RecoveryProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.calls += 1
            tool_calls = [
                ToolUseBlock(
                    id="call-write-before-read",
                    name="write_file",
                    input={
                        "file_path": "target.txt",
                        "content": "RECOVERED_WRITE_TOKEN_0708",
                    },
                ),
                ToolUseBlock(
                    id="call-read-after-error",
                    name="read_file",
                    input={"file_path": "target.txt"},
                ),
                ToolUseBlock(
                    id="call-write-after-read",
                    name="write_file",
                    input={
                        "file_path": "target.txt",
                        "content": "RECOVERED_WRITE_TOKEN_0708",
                    },
                ),
            ]
            return LLMResponse(
                provider="fake",
                id=f"recovery-{self.calls}",
                model=request.model,
                content=[tool_calls[self.calls - 1]],
                finish_reason="tool_calls",
            )

    monkeypatch.setattr(
        providers,
        "create_provider_client",
        lambda *args, **kwargs: RecoveryProvider(),
    )

    report = asyncio.run(module.run_file_not_read_recovery_live(tmp_path, retries=0))

    assert report["ok"] is True
    assert (
        report["actual"] == "write_file:FILE_NOT_READ -> read_file:ok -> write_file:ok"
    )
    assert report["api_calls"] == 3
    assert report["steps"] == 3
    assert report["evidence"]["tool_sequence"] == [
        "write_file",
        "read_file",
        "write_file",
    ]
    assert report["evidence"]["target_text"] == "RECOVERED_WRITE_TOKEN_0708"


def test_retry_assertion_aggregates_attempted_api_calls_and_steps() -> None:
    utils = _load_flow_module("deepseek_flow.utils")
    reporting = _load_flow_module("deepseek_flow.reporting")
    attempts = [
        reporting.scenario_report(
            name="runtime_read_loop_live",
            ok=False,
            status="assertion_failed",
            api_calls=2,
            steps=2,
            expected="tool call",
            actual="no tool call",
            evidence={},
            error_code="ASSERTION_FAILED",
            error_message="模型未按要求调用工具",
        ),
        reporting.scenario_report(
            name="runtime_read_loop_live",
            ok=True,
            status="ok",
            api_calls=1,
            steps=1,
            expected="tool call",
            actual="tool call",
            evidence={},
        ),
    ]

    async def attempt() -> dict[str, object]:
        return attempts.pop(0)

    report = asyncio.run(utils._retry_assertion(attempt, retries=2))

    assert report["ok"] is True
    assert report["api_calls"] == 3
    assert report["steps"] == 3
    assert report["evidence"]["attempt"] == 2
    assert report["evidence"]["attempts"] == [
        {
            "attempt": 1,
            "ok": False,
            "status": "assertion_failed",
            "api_calls": 2,
            "steps": 2,
            "actual": "no tool call",
            "error_code": "ASSERTION_FAILED",
            "error_message": "模型未按要求调用工具",
        },
        {
            "attempt": 2,
            "ok": True,
            "status": "ok",
            "api_calls": 1,
            "steps": 1,
            "actual": "tool call",
            "error_code": "",
            "error_message": "",
        },
    ]


def test_retry_assertion_does_not_retry_provider_errors() -> None:
    utils = _load_flow_module("deepseek_flow.utils")
    reporting = _load_flow_module("deepseek_flow.reporting")
    calls = 0

    async def attempt() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return reporting.scenario_report(
            name="runtime_read_loop_live",
            ok=False,
            status="error",
            api_calls=1,
            steps=1,
            expected="tool call",
            actual="provider failed",
            evidence={},
            error_code="PROVIDER_ERROR",
            error_message="Cannot connect",
        )

    report = asyncio.run(utils._retry_assertion(attempt, retries=2))

    assert calls == 1
    assert report["api_calls"] == 1
    assert report["steps"] == 1
    assert report["evidence"]["attempt"] == 1


def test_scenario_report_has_stable_json_safe_shape() -> None:
    module = _load_flow_module("deepseek_flow.reporting")

    report = module.scenario_report(
        name="provider_smoke_live",
        ok=True,
        status="ok",
        api_calls=1,
        steps=1,
        expected="IRIS_PROVIDER_OK",
        actual="IRIS_PROVIDER_OK",
        evidence={"provider": "deepseek"},
    )

    assert report == {
        "name": "provider_smoke_live",
        "ok": True,
        "status": "ok",
        "api_calls": 1,
        "steps": 1,
        "expected": "IRIS_PROVIDER_OK",
        "actual": "IRIS_PROVIDER_OK",
        "evidence": {"provider": "deepseek"},
        "error_code": "",
        "error_message": "",
    }


def test_aggregate_report_fails_when_any_scenario_fails(tmp_path: Path) -> None:
    module = _load_flow_module("deepseek_flow.reporting")
    passed = module.scenario_report(
        name="provider_smoke_live",
        ok=True,
        status="ok",
        api_calls=1,
        steps=1,
        expected="ok",
        actual="ok",
        evidence={},
    )
    failed = module.scenario_report(
        name="runtime_read_loop_live",
        ok=False,
        status="error",
        api_calls=1,
        steps=1,
        expected="tool call",
        actual="text only",
        evidence={},
        error_code="ASSERTION_FAILED",
        error_message="模型未按要求调用工具",
    )

    report = module.aggregate_report(
        tmp_path,
        [passed, failed],
        metadata={"scenario": "all", "log_dir": str(tmp_path / "logs")},
    )

    assert report["schema_version"] == 1
    assert report["ok"] is False
    assert report["work_dir"] == str(tmp_path)
    assert report["metadata"]["scenario"] == "all"
    assert report["metadata"]["log_dir"] == str(tmp_path / "logs")
    assert "runtime_read_loop_live" in report["scenario_catalog"]
    assert report["scenario_catalog"]["runtime_read_loop_live"]["module"] == "runtime"
    assert report["scenario_count"] == 2
    assert report["total_api_calls"] == 2
    assert report["total_steps"] == 2
    assert report["failed_scenarios"] == ["runtime_read_loop_live"]
    assert report["blocking_scenarios"] == ["runtime_read_loop_live"]
    assert report["blocking_modules"] == ["runtime"]
    assert report["module_coverage"] == [
        {
            "module": "providers",
            "ok": True,
            "scenario_count": 1,
            "failed_scenarios": [],
            "api_calls": 1,
            "steps": 1,
            "runtime_apis": ["direct_provider_call"],
        },
        {
            "module": "runtime",
            "ok": False,
            "scenario_count": 1,
            "failed_scenarios": ["runtime_read_loop_live"],
            "api_calls": 1,
            "steps": 1,
            "runtime_apis": ["run_loop"],
        },
    ]
    assert report["failure_summary"] == [
        {
            "name": "runtime_read_loop_live",
            "status": "error",
            "error_code": "ASSERTION_FAILED",
            "error_message": "模型未按要求调用工具",
            "actual": "text only",
        }
    ]


def test_run_selected_scenarios_records_scenario_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_flow_module("deepseek_flow.runner")
    reporting = _load_flow_module("deepseek_flow.reporting")

    async def runner(work_dir: Path, retries: int) -> dict[str, object]:
        return reporting.scenario_report(
            name="provider_smoke_live",
            ok=True,
            status="ok",
            api_calls=1,
            steps=1,
            expected="ok",
            actual="ok",
            evidence={"work_dir_seen": str(work_dir), "retries_seen": retries},
        )

    monkeypatch.setitem(module.SCENARIO_RUNNERS, "provider_smoke_live", runner)

    reports = asyncio.run(
        module.run_selected_scenarios(
            tmp_path,
            scenario="provider_smoke_live",
            retries=3,
        )
    )

    assert reports[0]["scenario_dir"] == str(tmp_path / "provider_smoke_live")
    assert reports[0]["module"] == "providers"
    assert reports[0]["runtime_api"] == "direct_provider_call"
    assert reports[0]["evidence"]["work_dir_seen"] == reports[0]["scenario_dir"]
    assert reports[0]["evidence"]["retries_seen"] == 3


def test_run_deepseek_flow_records_environment_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_flow_module("deepseek_flow.runner")
    reporting = _load_flow_module("deepseek_flow.reporting")

    async def fake_run_selected_scenarios(
        work_dir: Path,
        *,
        scenario: str,
        retries: int,
    ) -> list[dict[str, object]]:
        assert work_dir == tmp_path
        assert scenario == "provider_smoke_live"
        assert retries == 0
        return [
            reporting.scenario_report(
                name="provider_smoke_live",
                ok=True,
                status="ok",
                api_calls=1,
                steps=1,
                expected="ok",
                actual="ok",
                evidence={},
            )
        ]

    monkeypatch.setattr(module, "run_selected_scenarios", fake_run_selected_scenarios)
    monkeypatch.setattr(module, "resolve_api_key", lambda: "sk-test")
    monkeypatch.setattr(
        module,
        "collect_run_environment",
        lambda root: {"repo_root": str(root), "git": {"commit": "abc123"}},
    )

    report = asyncio.run(
        module.run_deepseek_flow(
            work_dir=tmp_path,
            scenario="provider_smoke_live",
            retries=0,
            log_dir=tmp_path / "logs",
        )
    )

    assert report["metadata"]["environment"]["git"]["commit"] == "abc123"
    assert report["metadata"]["environment"]["repo_root"]


def test_collect_run_environment_records_python_and_git_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_flow_module("deepseek_flow.environment")
    commands: list[tuple[str, ...]] = []

    class Completed:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> Completed:
        commands.append(tuple(command))
        assert cwd == tmp_path
        assert check is False
        assert capture_output is True
        assert text is True
        match command[1:]:
            case ["rev-parse", "HEAD"]:
                return Completed("abc123\n")
            case ["branch", "--show-current"]:
                return Completed("codex/test\n")
            case ["status", "--porcelain"]:
                return Completed(" M scripts/deepseek_agent_flow.py\n")
            case _:
                raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.platform, "platform", lambda: "test-platform")

    metadata = module.collect_run_environment(tmp_path)

    assert metadata["repo_root"] == str(tmp_path)
    assert metadata["python_version"]
    assert metadata["python_executable"]
    assert metadata["platform"] == "test-platform"
    assert metadata["git"] == {
        "commit": "abc123",
        "branch": "codex/test",
        "dirty": True,
        "status_short": ["M scripts/deepseek_agent_flow.py"],
    }
    assert commands == [
        ("git", "rev-parse", "HEAD"),
        ("git", "branch", "--show-current"),
        ("git", "status", "--porcelain"),
    ]


def test_write_report_persists_json_and_records_path(tmp_path: Path) -> None:
    module = _load_flow_module("deepseek_flow.reporting")
    report = module.aggregate_report(tmp_path, [])

    report_path = module.write_report(tmp_path, report)

    assert report_path == tmp_path / "report.json"
    assert report["report_path"] == str(report_path)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["report_path"] == str(report_path)
    assert persisted["schema_version"] == 1
    assert persisted["scenarios"] == []


def test_write_report_persists_human_summary(tmp_path: Path) -> None:
    module = _load_flow_module("deepseek_flow.reporting")
    failed = module.scenario_report(
        name="runtime_read_loop_live",
        ok=False,
        status="error",
        api_calls=2,
        steps=2,
        expected="IRIS_RUNTIME_TOOL_OK",
        actual="missing prefix",
        evidence={},
        error_code="ASSERTION_FAILED",
        error_message="最终回答缺少前缀",
    )
    failed["scenario_dir"] = str(tmp_path / "runtime_read_loop_live")
    report = module.aggregate_report(
        tmp_path,
        [failed],
        metadata={
            "environment": {
                "git": {
                    "commit": "abc123",
                    "branch": "codex/test",
                    "dirty": True,
                }
            }
        },
    )

    module.write_report(tmp_path, report)

    summary_path = tmp_path / "summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    assert report["summary_path"] == str(summary_path)
    assert "# DeepSeek Flow Summary" in summary
    assert "- Result: FAIL" in summary
    assert "- Scenarios: 1" in summary
    assert "- Total API calls: 2" in summary
    assert "- Git commit: abc123" in summary
    assert "- Git branch: codex/test" in summary
    assert "- Git dirty: True" in summary
    assert "## Module Coverage" in summary
    assert (
        "| Module | Result | Scenarios | API calls | Steps | Runtime APIs | Failed scenarios |"
        in summary
    )
    assert (
        "| runtime | FAIL | 1 | 2 | 2 | run_loop | runtime_read_loop_live |" in summary
    )
    scenario_matrix_header = (
        "| Scenario | Module | Runtime API | Result | Status | "
        "API calls | Steps | Scenario dir | Error |"
    )
    assert scenario_matrix_header in summary
    assert "runtime_read_loop_live" in summary
    assert "runtime" in summary
    assert "run_loop" in summary
    assert "ASSERTION_FAILED" in summary
    assert str(tmp_path / "runtime_read_loop_live") in summary
    assert str(tmp_path / "report.json") in summary


def test_amain_returns_2_when_live_key_is_missing(tmp_path: Path) -> None:
    module = _load_flow_module("deepseek_flow.cli")
    module.init_local_config = lambda base_dir: False
    module.resolve_api_key = lambda: None

    exit_code = asyncio.run(module.amain(["--work-dir", str(tmp_path)]))

    assert exit_code == 2


def test_setup_flow_logging_uses_iris_log_setup_logger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_flow_module("deepseek_flow.config")
    log_dir = tmp_path / "logs"
    calls: list[Path] = []

    def fake_setup_logger(log_dir: Path) -> None:
        calls.append(log_dir)

    monkeypatch.setattr(module, "setup_logger", fake_setup_logger)

    module.setup_flow_logging(log_dir)

    assert calls == [log_dir]


def test_legacy_script_reexports_public_entrypoints() -> None:
    module = _load_demo_module()

    assert module.SCENARIO_NAMES
    assert module.parse_args([]).scenario == "all"
    assert callable(module.main)
    assert callable(module.write_report)


def test_live_deepseek_gate_covers_runtime_and_tool_paths() -> None:
    expected = {
        "provider_smoke_live",
        "runtime_read_loop_live",
        "builtin_file_tools_live",
        "file_not_read_recovery_live",
    }

    assert expected.issubset(set(LIVE_DEEPSEEK_GATE_SCENARIOS))


@pytest.mark.live_deepseek
@pytest.mark.parametrize("scenario", LIVE_DEEPSEEK_GATE_SCENARIOS)
def test_live_deepseek_script_runs_selected_scenario(
    tmp_path: Path,
    pytestconfig: pytest.Config,
    scenario: str,
) -> None:
    """显式启用时，通过真实 DeepSeek API 跑脚本入口。"""
    if not pytestconfig.getoption("--run-live-deepseek"):
        pytest.skip("需要显式传入 --run-live-deepseek 才运行真实 DeepSeek API 测试")
    config_module = _load_flow_module("deepseek_flow.config")
    repo_root = Path(__file__).resolve().parents[2]
    config_module.init_local_config(repo_root)
    if config_module.resolve_api_key() is None:
        pytest.skip("缺少 IRIS_PROVIDER_API_KEYS__DEEPSEEK 或 IRIS_API_KEY")

    script_path = repo_root / "scripts" / "deepseek_agent_flow.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--scenario",
            scenario,
            "--work-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["scenarios"][0]["name"] == scenario
    assert report["scenarios"][0]["api_calls"] >= 1
