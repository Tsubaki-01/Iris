"""Catalog 的唯一解析边界与懒加载契约。"""

from pathlib import Path

import pytest
import yaml

from iris.agents.config.subagent import load_subagent_catalog
from iris.exceptions import IrisConfigError


def test_catalog_resolves_child_paths_without_loading_children(tmp_path: Path) -> None:
    child = tmp_path / "broken.yaml"
    child.write_text("[invalid yaml", encoding="utf-8")
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "default: researcher\nagents:\n"
        "  researcher:\n    path: missing/agent.yaml\n    description: '  Search notes  '\n"
        "  reviewer:\n    path: broken.yaml\n    description: Review notes\n",
        encoding="utf-8",
    )
    table = load_subagent_catalog(path)
    assert table.default == "researcher"
    assert table.routes["researcher"].config_path == tmp_path / "missing" / "agent.yaml"
    assert table.routes["researcher"].description == "Search notes"
    assert table.routes["reviewer"].config_path == child
    path.write_text("{}", encoding="utf-8")
    assert list(table.routes) == ["researcher", "reviewer"]
    with pytest.raises(TypeError):
        table.routes["new"] = table.routes["researcher"]  # type: ignore[index]


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"default": "a", "agents": {}},
        {"default": "b", "agents": {"a": {"path": "x", "description": "A"}}},
        {"default": "a", "agents": {"a": {"path": "x", "description": " "}}},
        {"default": "a", "agents": {"a": {"path": "x", "description": "A", "extra": 1}}},
        {"default": "a", "agents": {"a": {"path": "x", "description": "A"}}, "extra": 1},
        *[
            {"default": key, "agents": {key: {"path": "x", "description": "A"}}}
            for key in ["A", " a", "a ", "a_b", "a--b"]
        ],
    ],
)
def test_catalog_rejects_invalid_raw_config(tmp_path: Path, raw: object) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(IrisConfigError) as caught:
        load_subagent_catalog(path)
    assert caught.value.context["path"] == str(path)


@pytest.mark.parametrize("content", [None, b"\xff", b"[invalid"])
def test_catalog_wraps_file_encoding_and_yaml_errors(tmp_path: Path, content: bytes | None) -> None:
    path = tmp_path / "catalog.yaml"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(IrisConfigError) as caught:
        load_subagent_catalog(path)
    assert caught.value.context["path"] == str(path)
