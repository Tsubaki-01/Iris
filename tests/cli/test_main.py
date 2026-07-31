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


def test_main_dispatches_run_events(monkeypatch) -> None:
    captured: dict[str, object] = {}
    cli_main = importlib.import_module("iris.cli.main")

    def fake_run_events(options: object) -> int:
        captured["options"] = options
        return 11

    monkeypatch.setattr(cli_main, "run_events", fake_run_events, raising=False)

    code = cli_main.main(
        [
            "run",
            "events",
            "agent.yaml",
            "--run-id",
            "run-1",
            "--after-sequence",
            "7",
            "--env-file",
            ".env.local",
            "--json",
        ]
    )

    assert code == 11
    options = captured["options"]
    assert options.__class__.__name__ == "RunEventsOptions"
    assert options.config_path == Path("agent.yaml")
    assert options.run_id == "run-1"
    assert options.after_sequence == 7
    assert options.env_file == Path(".env.local")
    assert options.json_output is True
