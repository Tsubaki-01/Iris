"""使用 mock LLM provider 验证 Iris 当前 Agent runtime 主流程。

直接运行本脚本不会访问真实模型服务。脚本会创建临时 `agent.yaml`、workspace 文件和
内存 session，通过 `RuntimeFactory` 装配 runtime，并输出各场景的诊断报告。

Example:
    uv run python scripts/mock_agent_flow.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from iris.memory import (  # noqa: E402
    MemoryCategory,
    MemoryItem,
    MemoryItemKind,
    MemoryLevel,
    MemoryScope,
    MemorySearchResult,
)
from iris.message import LLMRequest, LLMResponse, TextBlock, ToolUseBlock  # noqa: E402
from iris.runtime import RuntimeFactory  # noqa: E402
from iris.runtime.models import BoundedLoopOptions, RuntimeOptions  # noqa: E402


class MockLLMProvider:
    """按顺序返回预置响应，并记录 runtime 发出的 `LLMRequest`。"""

    def __init__(self, responses: list[LLMResponse]) -> None:
        """创建 mock provider。

        Args:
            responses: 按调用顺序返回的 provider-neutral 响应。
        """
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """记录请求并返回下一条 mock 响应。"""
        self.requests.append(request)
        if not self._responses:
            return _assistant_text_response("mock provider 没有更多响应。")
        return self._responses.pop(0)


def run_demo(base_dir: Path | None = None) -> dict[str, Any]:
    """运行全部 mock agent flow 场景并返回结构化诊断报告。

    Args:
        base_dir: 可选工作目录。未传入时使用临时目录，脚本结束后自动清理。

    Returns:
        包含场景结果、发现问题和改正方案的 JSON-safe 字典。
    """
    if base_dir is not None:
        return asyncio.run(_run_demo(base_dir))

    with TemporaryDirectory(prefix="iris-mock-agent-flow-") as tmp_dir:
        return asyncio.run(_run_demo(Path(tmp_dir)))


async def _run_demo(base_dir: Path) -> dict[str, Any]:
    """异步运行示例场景。"""
    workspace = _prepare_workspace(base_dir)
    agent_path = _write_agent_yaml(base_dir)

    tool_loop = await _run_tool_loop(agent_path)
    single_turn = await _run_single_turn_tool_bridge(agent_path)
    max_steps = await _run_max_steps_probe(agent_path)
    explicit_memory = await _run_explicit_memory(base_dir)

    scenarios = {
        "tool_loop": tool_loop,
        "single_turn_tool_bridge": single_turn,
        "max_steps_probe": max_steps,
        "explicit_memory": explicit_memory,
    }
    failed_scenarios = [
        name for name, scenario in scenarios.items() if scenario["status"] == "error"
    ]
    return {
        "workspace": str(workspace),
        "summary": {
            "scenario_count": len(scenarios),
            "failed_scenarios": failed_scenarios,
        },
        "scenarios": scenarios,
        "findings": _findings(single_turn, max_steps),
    }


def _prepare_workspace(base_dir: Path) -> Path:
    """创建示例 workspace 和可被文件工具读取的文本文件。"""
    workspace = base_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "notes.txt").write_text(
        "Iris 是一个本地优先 Agent Kit。\n",
        encoding="utf-8",
    )
    return workspace


def _write_agent_yaml(base_dir: Path) -> Path:
    """写入覆盖 factory、context、工具和 session 配置的 agent YAML。"""
    agent_path = base_dir / "agent.yaml"
    agent_path.write_text(
        """
name: mock-file-agent
model:
  provider: openai
  name: gpt-4o-mini
  api_style: responses
system: |
  你是 Iris 示例助手。需要读取文件时使用 read_file。
tools:
  builtin:
    - file.read
permissions:
  workspace: workspace
  writes: deny
session:
  backend: none
""".strip(),
        encoding="utf-8",
    )
    return agent_path


async def _run_tool_loop(agent_path: Path) -> dict[str, Any]:
    """验证 `run_loop()` 会把工具结果回灌到下一次 provider 请求。"""
    provider = MockLLMProvider(
        [
            _assistant_tool_response("call_read_loop"),
            _assistant_text_response("notes.txt 说明 Iris 是一个本地优先 Agent Kit。"),
        ]
    )
    runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
    result = await runtime.run_loop(
        "读取 notes.txt 并总结。",
        options=RuntimeOptions(
            session_id="tool-loop",
            loop=BoundedLoopOptions(max_steps=3),
            metadata={"scenario": "tool_loop"},
        ),
    )
    second_request = provider.requests[1] if len(provider.requests) > 1 else None
    metadata = runtime.session_store.load_run_metadata("tool-loop")
    return {
        "status": result.status.value,
        "steps": result.steps,
        "provider_request_count": len(provider.requests),
        "first_request_tool_schema_count": len(provider.requests[0].tools),
        "second_request_has_tool_result": _request_has_tool_result(second_request),
        "tool_results": [
            tool_result.model_content for tool_result in result.tool_results
        ],
        "tool_events": runtime.session_store.load_tool_events("tool-loop"),
        "latest_run": metadata.get("latest_run", {}),
    }


async def _run_single_turn_tool_bridge(agent_path: Path) -> dict[str, Any]:
    """验证 `run_turn()` 只执行一次工具桥接，不发起第二次 provider 调用。"""
    provider = MockLLMProvider([_assistant_tool_response("call_read_turn")])
    runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
    result = await runtime.run_turn(
        "读取 notes.txt。",
        options=RuntimeOptions(
            session_id="single-turn",
            metadata={"scenario": "single_turn_tool_bridge"},
        ),
    )
    return {
        "status": result.status.value,
        "provider_request_count": len(provider.requests),
        "tool_result_count": len(result.tool_results),
        "tool_results": [
            tool_result.model_content for tool_result in result.tool_results
        ],
    }


async def _run_max_steps_probe(agent_path: Path) -> dict[str, Any]:
    """验证 provider 持续返回工具调用时，bounded loop 会在步数上限停止。"""
    provider = MockLLMProvider([_assistant_tool_response("call_read_max")])
    runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
    result = await runtime.run_loop(
        "一直读取 notes.txt。",
        options=RuntimeOptions(
            session_id="max-steps",
            loop=BoundedLoopOptions(max_steps=1),
            metadata={"scenario": "max_steps_probe"},
        ),
    )
    return {
        "status": result.status.value,
        "steps": result.steps,
        "provider_request_count": len(provider.requests),
        "error_code": result.error.code if result.error is not None else "",
        "error_message": result.error.message if result.error is not None else "",
    }


async def _run_explicit_memory(base_dir: Path) -> dict[str, Any]:
    """验证显式 memory_results 会进入 context，且召回分数不会泄漏进 prompt。"""
    agent_path = base_dir / "memory-agent.yaml"
    agent_path.write_text(
        """
name: mock-memory-agent
model: openai/gpt-4o-mini
system: 你是 Iris 记忆示例助手。
session:
  backend: none
""".strip(),
        encoding="utf-8",
    )
    provider = MockLLMProvider([_assistant_text_response("已参考记忆。")])
    runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
    result = await runtime.run_turn(
        "我偏好什么回答风格？",
        options=RuntimeOptions(
            session_id="explicit-memory",
            memory_results=[_memory_result()],
            memory_max_chars=200,
            metadata={"scenario": "explicit_memory"},
        ),
    )
    prompt_text = "\n".join(message.text for message in provider.requests[0].messages)
    return {
        "status": result.status.value,
        "provider_request_count": len(provider.requests),
        "memory_context_in_request": "<memory_context>" in prompt_text,
        "score_leaked_to_prompt": "0.98" in prompt_text or "sqlite" in prompt_text,
    }


def _assistant_tool_response(call_id: str) -> LLMResponse:
    """构造要求读取 `notes.txt` 的 assistant 工具调用响应。"""
    return LLMResponse(
        provider="mock",
        id=f"response-{call_id}",
        model="gpt-4o-mini",
        content=[
            TextBlock(text="我需要读取示例文件。"),
            ToolUseBlock(
                id=call_id,
                name="read_file",
                input={"file_path": "notes.txt"},
            ),
        ],
        finish_reason="tool_calls",
        input_tokens=10,
        output_tokens=8,
        total_tokens=18,
    )


def _assistant_text_response(text: str) -> LLMResponse:
    """构造普通文本 assistant 响应。"""
    return LLMResponse(
        provider="mock",
        id=f"response-text-{abs(hash(text))}",
        model="gpt-4o-mini",
        content=[TextBlock(text=text)],
        finish_reason="stop",
        input_tokens=10,
        output_tokens=8,
        total_tokens=18,
    )


def _memory_result() -> MemorySearchResult:
    """构造显式 memory 结果，供 runtime 注入 context。"""
    item = MemoryItem(
        id="memory-1",
        scope=MemoryScope(workspace_id="workspace", agent_id="mock-memory-agent"),
        text="用户喜欢简洁中文回答。",
        category=MemoryCategory.USER,
        kind=MemoryItemKind.PREFERENCE,
        level=MemoryLevel.SEMANTIC,
    )
    return MemorySearchResult(
        item=item,
        score=0.98,
        source="sqlite",
        matched_text=item.text,
    )


def _request_has_tool_result(request: LLMRequest | None) -> bool:
    """判断 provider 请求中是否包含工具结果消息。"""
    if request is None:
        return False
    return any(message.tool_results for message in request.messages)


def _findings(
    single_turn: dict[str, Any],
    max_steps: dict[str, Any],
) -> list[dict[str, str]]:
    """基于实际场景结果生成问题与改正方案。"""
    return [
        {
            "issue": "run_turn 会执行一次工具桥接，但不会把工具结果再次发送给 provider。",
            "evidence": (
                "single_turn_tool_bridge provider_request_count="
                f"{single_turn['provider_request_count']} 且 "
                f"tool_result_count={single_turn['tool_result_count']}。"
            ),
            "correction": (
                "需要让模型基于工具结果生成最终回答时，调用 run_loop() "
                "而不是 run_turn()。"
            ),
        },
        {
            "issue": "如果 mock/provider 持续返回工具调用，bounded loop 会在上限处停止。",
            "evidence": f"max_steps_probe 返回 {max_steps['error_code']}。",
            "correction": (
                "调大 RuntimeOptions.loop.max_steps，或改进模型指令让工具结果足够时"
                "输出最终文本。"
            ),
        },
    ]


def main() -> None:
    """运行示例并把报告打印为 JSON。"""
    report = run_demo()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
