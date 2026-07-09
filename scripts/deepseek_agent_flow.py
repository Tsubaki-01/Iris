"""使用真实 DeepSeek API 验证 Iris 当前 provider 与 runtime 全流程。

脚本会通过 Iris 集中配置读取本地 `.env.local` / `.env` 中的
`IRIS_PROVIDER_API_KEYS__DEEPSEEK` 或 `IRIS_API_KEY`，然后在临时 workspace
内运行一组 live integration 场景。所有场景都会至少发起一次真实 DeepSeek
API 调用；需要检查 request 形状时，通过 `RecordingRuntimeProvider` 包装真实
provider，不使用 fake provider 伪造 runtime 行为。

Example:
    uv run python scripts/deepseek_agent_flow.py
    uv run python scripts/deepseek_agent_flow.py --scenario builtin_file_tools_live
"""

from __future__ import annotations

from deepseek_flow import (
    SCENARIO_NAMES,
    RecordingRuntimeProvider,
    _provider_smoke_ok,
    _runtime_final_ok,
    _safe_error_message,
    aggregate_report,
    amain,
    init_local_config,
    main,
    parse_args,
    resolve_api_key,
    run_deepseek_flow,
    run_selected_scenarios,
    scenario_report,
    setup_flow_logging,
    write_report,
)

__all__ = [
    "RecordingRuntimeProvider",
    "SCENARIO_NAMES",
    "_provider_smoke_ok",
    "_runtime_final_ok",
    "_safe_error_message",
    "aggregate_report",
    "amain",
    "init_local_config",
    "main",
    "parse_args",
    "resolve_api_key",
    "run_deepseek_flow",
    "run_selected_scenarios",
    "scenario_report",
    "setup_flow_logging",
    "write_report",
]


if __name__ == "__main__":
    main()
