import importlib
import sys

from pathlib import Path

import pytest
from loguru import logger


def test_import_log_does_not_create_default_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("iris.log", None)

    importlib.import_module("iris.log")

    assert not (tmp_path / "iris_log").exists()


def test_setup_logger_with_log_dir_creates_file_sinks(tmp_path: Path) -> None:
    from iris.log import setup_logger

    log_dir = tmp_path / "logs"

    setup_logger(log_dir=log_dir)

    assert log_dir.is_dir()


def test_setup_logger_does_not_remove_external_sink() -> None:
    from iris.log import setup_logger

    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")

    try:
        setup_logger(log_dir=None)
        logger.info("external-check")
    finally:
        try:
            logger.remove(sink_id)
        except ValueError:
            pass

    assert "external-check\n" in messages
