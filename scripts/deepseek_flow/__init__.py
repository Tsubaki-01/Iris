"""DeepSeek live integration 验证脚本包。"""

from __future__ import annotations

from . import bootstrap as bootstrap
from .cli import amain, main, parse_args
from .config import (
    _safe_error_message,
    init_local_config,
    resolve_api_key,
    setup_flow_logging,
)
from .constants import SCENARIO_NAMES
from .providers import RecordingRuntimeProvider
from .reporting import aggregate_report, scenario_report, write_report
from .runner import run_deepseek_flow, run_selected_scenarios
from .utils import _provider_smoke_ok, _runtime_final_ok

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
