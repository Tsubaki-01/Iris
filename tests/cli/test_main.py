from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from iris.cli.chat import ChatOptions


def test_main_dispatches_only_chat(monkeypatch: pytest.MonkeyPatch) -> None:
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
            "--env-file",
            ".env.local",
            "--no-tools",
        ]
    )
    assert code == 7
    assert captured["options"] == ChatOptions(
        config_path=Path("agent.yaml"),
        session_id="demo",
        max_steps=4,
        env_file=Path(".env.local"),
        include_tools=False,
    )


def test_main_rejects_removed_run_command() -> None:
    cli_main = importlib.import_module("iris.cli.main")
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["run", "events", "agent.yaml", "--run-id", "run-1"])
    assert exc_info.value.code == 2


def test_main_without_command_prints_help_and_returns_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli_main = importlib.import_module("iris.cli.main")
    monkeypatch.setattr(sys, "argv", ["iris"])

    assert cli_main.main(None) == 1
    assert "chat" in capsys.readouterr().out
