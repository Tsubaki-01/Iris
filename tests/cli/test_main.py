from __future__ import annotations

import importlib
import io
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


def test_main_prints_unicode_to_utf8_when_stdout_defaults_to_gbk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式 CLI 在启动 chat 前配置输出，避免网页字符令 GBK 管道崩溃。"""
    cli_main = importlib.import_module("iris.cli.main")
    buffer = io.BytesIO()
    output = io.TextIOWrapper(buffer, encoding="gbk")

    def fake_run_chat(options: ChatOptions) -> int:
        """通过与 chat 默认回调相同的标准输出打印模型文本。"""
        print("Python 文档 ¶ 😀", end="", flush=True)
        return 0

    with monkeypatch.context() as patch:
        patch.setattr(cli_main, "run_chat", fake_run_chat)
        patch.setattr(sys, "stdout", output)
        code = cli_main.main(["chat", "agent.yaml"])

    assert code == 0
    assert buffer.getvalue().decode("utf-8") == "Python 文档 ¶ 😀"
