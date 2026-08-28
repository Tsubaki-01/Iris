from pathlib import Path

from iris.skill._paths import is_resolved_within


def test_is_resolved_within_accepts_boundary_and_descendant(tmp_path: Path) -> None:
    boundary = tmp_path.resolve()

    assert is_resolved_within(boundary, boundary) is True
    assert is_resolved_within(boundary / "skill" / "SKILL.md", boundary) is True


def test_is_resolved_within_rejects_sibling(tmp_path: Path) -> None:
    boundary = (tmp_path / "workspace").resolve()
    sibling = (tmp_path / "outside" / "SKILL.md").resolve()

    assert is_resolved_within(sibling, boundary) is False
