from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path
from types import ModuleType

import pytest

from iris.message import LLMRequest, LLMResponse, TextBlock, ToolUseBlock


class FakeProvider:
    """测试用 provider，按顺序返回预置响应并记录请求。"""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _load_demo_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "deepseek_agent_flow.py"
    spec = importlib.util.spec_from_file_location("deepseek_agent_flow", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_local_env_reads_deepseek_key_without_overriding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_demo_module()
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "IRIS_DEEPSEEK_API_KEY=local-key\nIRIS_OTHER=value\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("IRIS_DEEPSEEK_API_KEY", "existing-key")
    monkeypatch.delenv("IRIS_OTHER", raising=False)

    loaded = module.load_local_env(tmp_path)

    assert loaded == {"IRIS_OTHER": "value"}
    assert module.resolve_api_key() == "existing-key"


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
            api_key="test-key",
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
            api_key="test-key",
            expected_token=expected_token,
            provider=provider,
        )
    )
    for handler in module.LOGGER.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()

    content = (log_dir / "runtime.log").read_text(encoding="utf-8")

    assert "deepseek.runtime.start" in content
    assert "agent_path=" in content
    assert "deepseek.runtime.finish" in content
    assert "status=ok" in content
    assert "tool_result_count=1" in content
    assert "expected_token_found=True" in content
