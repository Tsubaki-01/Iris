from __future__ import annotations

import asyncio
import importlib.util
import inspect
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import pytest

import iris
from iris.message import LLMRequest, LLMResponse, TextBlock, ToolUseBlock


@pytest.fixture(autouse=True)
def reset_config_state(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """隔离脚本测试中的全局配置状态。"""
    for name in (
        "IRIS_API_KEY",
        "IRIS_PROVIDER_API_KEYS__DEEPSEEK",
    ):
        monkeypatch.delenv(name, raising=False)
    iris.reset()
    yield
    iris.reset()


class FakeProvider:
    """测试用 provider，按顺序返回预置响应并记录请求。"""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.responses.pop(0)


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


def test_init_local_config_loads_api_key_without_overriding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_demo_module()
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
    module = _load_demo_module()
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


def test_deepseek_runtime_flow_uses_tool_loop_with_injected_provider(
    tmp_path: Path,
) -> None:
    module = _load_demo_module()
    agent_path, expected_token = module.prepare_runtime_workspace(tmp_path)
    provider = FakeProvider(
        [
            LLMResponse(
                provider="fake",
                id="tool-call",
                model="deepseek-chat",
                content=[
                    ToolUseBlock(
                        id="call-read",
                        name="read_file",
                        input={"file_path": "verification.txt"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                provider="fake",
                id="final",
                model="deepseek-chat",
                content=[TextBlock(text=f"IRIS_RUNTIME_TOOL_OK: {expected_token}")],
                finish_reason="stop",
            ),
        ]
    )

    report = asyncio.run(
        module.run_runtime_tool_loop(
            agent_path=agent_path,
            expected_token=expected_token,
            provider=provider,
        )
    )

    assert report["ok"] is True
    assert report["status"] == "ok"
    assert report["provider_request_count"] == 2
    assert report["tool_result_count"] == 1
    assert report["expected_token_found"] is True
    assert provider.requests[0].tools
    assert any(message.tool_results for message in provider.requests[1].messages)


def test_deepseek_runtime_flow_logs_important_data_nodes(tmp_path: Path) -> None:
    module = _load_demo_module()
    agent_path, expected_token = module.prepare_runtime_workspace(tmp_path)
    log_dir = tmp_path / "logs"
    module.setup_flow_logging(log_dir)
    provider = FakeProvider(
        [
            LLMResponse(
                provider="fake",
                id="tool-call",
                model="deepseek-chat",
                content=[
                    ToolUseBlock(
                        id="call-read",
                        name="read_file",
                        input={"file_path": "verification.txt"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                provider="fake",
                id="final",
                model="deepseek-chat",
                content=[TextBlock(text=f"IRIS_RUNTIME_TOOL_OK: {expected_token}")],
                finish_reason="stop",
            ),
        ]
    )

    asyncio.run(
        module.run_runtime_tool_loop(
            agent_path=agent_path,
            expected_token=expected_token,
            provider=provider,
        )
    )
    complete_result = module.logger.complete()
    if inspect.isawaitable(complete_result):

        async def wait_for_logs() -> None:
            await complete_result

        asyncio.run(wait_for_logs())

    content = (log_dir / "runtime.log").read_text(encoding="utf-8")

    assert "deepseek.runtime.start" in content
    assert "agent_path=" in content
    assert "deepseek.runtime.finish" in content
    assert "status=ok" in content
    assert "tool_result_count=1" in content
    assert "expected_token_found=True" in content


def test_setup_flow_logging_uses_iris_log_setup_logger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_demo_module()
    log_dir = tmp_path / "logs"
    calls: list[Path] = []

    def fake_setup_logger(log_dir: Path) -> None:
        calls.append(log_dir)

    monkeypatch.setattr(module, "setup_logger", fake_setup_logger)

    module.setup_flow_logging(log_dir)

    assert calls == [log_dir]
