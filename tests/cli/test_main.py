from __future__ import annotations

import importlib
from pathlib import Path

from iris.cli.chat import ChatOptions


def test_main_dispatches_chat(monkeypatch) -> None:
    captured: dict[str, ChatOptions] = {}
    cli_main = importlib.import_module("iris.cli.main")

    def fake_run_chat(options: ChatOptions) -> int:
        captured["options"] = options
        return 7

    monkeypatch.setattr(cli_main, "run_chat", fake_run_chat)

    code = cli_main.main(
        [
            "chat",
            "agent.yaml",
            "--session-id",
            "demo",
            "--max-steps",
            "4",
            "--trace",
            "full",
            "--trace-file",
            "trace.jsonl",
            "--env-file",
            ".env.local",
            "--no-tools",
        ]
    )

    assert code == 7
    options = captured["options"]
    assert options.config_path == Path("agent.yaml")
    assert options.session_id == "demo"
    assert options.max_steps == 4
    assert options.trace_mode == "full"
    assert options.trace_file == Path("trace.jsonl")
    assert options.env_file == Path(".env.local")
    assert options.include_tools is False
