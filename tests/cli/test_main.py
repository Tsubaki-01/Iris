from __future__ import annotations

import importlib
from pathlib import Path


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
