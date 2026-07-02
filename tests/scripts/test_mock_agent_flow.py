from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_demo_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "mock_agent_flow.py"
    spec = importlib.util.spec_from_file_location("mock_agent_flow", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mock_agent_flow_reports_runtime_capabilities(tmp_path: Path) -> None:
    module = _load_demo_module()

    report = module.run_demo(tmp_path)

    assert report["summary"]["scenario_count"] == 4
    assert report["summary"]["failed_scenarios"] == []

    loop = report["scenarios"]["tool_loop"]
    assert loop["status"] == "ok"
    assert loop["provider_request_count"] == 2
    assert loop["tool_results"] == ["1: Iris 是一个本地优先 Agent Kit。"]
    assert loop["second_request_has_tool_result"] is True
    assert loop["latest_run"]["status"] == "ok"
    assert loop["latest_run"]["tool_count"] == 1
    assert loop["tool_events"][0]["tool_name"] == "read_file"

    run_turn = report["scenarios"]["single_turn_tool_bridge"]
    assert run_turn["status"] == "ok"
    assert run_turn["provider_request_count"] == 1
    assert run_turn["tool_result_count"] == 1

    max_steps = report["scenarios"]["max_steps_probe"]
    assert max_steps["status"] == "max_steps"
    assert max_steps["error_code"] == "MAX_STEPS_REACHED"
    assert max_steps["steps"] == 1

    memory = report["scenarios"]["explicit_memory"]
    assert memory["status"] == "ok"
    assert memory["memory_context_in_request"] is True
    assert memory["score_leaked_to_prompt"] is False

    assert report["findings"] == [
        {
            "issue": "run_turn 会执行一次工具桥接，但不会把工具结果再次发送给 provider。",
            "evidence": "single_turn_tool_bridge provider_request_count=1 且 tool_result_count=1。",
            "correction": (
                "需要让模型基于工具结果生成最终回答时，调用 run_loop() "
                "而不是 run_turn()。"
            ),
        },
        {
            "issue": "如果 mock/provider 持续返回工具调用，bounded loop 会在上限处停止。",
            "evidence": "max_steps_probe 返回 MAX_STEPS_REACHED。",
            "correction": (
                "调大 RuntimeOptions.loop.max_steps，或改进模型指令让工具结果足够时"
                "输出最终文本。"
            ),
        },
    ]
