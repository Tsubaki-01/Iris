from __future__ import annotations

import pytest
import yaml

from iris.exceptions import IrisSkillFormatError
from iris.skill.frontmatter import parse_frontmatter, split_frontmatter


def test_split_and_parse_valid_frontmatter_without_consuming_body() -> None:
    text = "---\ndescription: Example skill\nname: declared-name\n---\n# Body\nDetails\n"

    frontmatter, body = split_frontmatter(text)
    parsed = parse_frontmatter(frontmatter)

    assert frontmatter == "description: Example skill\nname: declared-name\n"
    assert body == "# Body\nDetails\n"
    assert parsed == {"description": "Example skill", "name": "declared-name"}
    assert "Body" not in parsed


def test_opening_delimiter_without_closing_delimiter_is_rejected() -> None:
    with pytest.raises(
        IrisSkillFormatError,
        match="SKILL.md frontmatter 缺少闭合分隔符",
    ):
        split_frontmatter("---\ndescription: Example skill\n# Body\n")


def test_malformed_yaml_is_wrapped_as_skill_format_error() -> None:
    with pytest.raises(
        IrisSkillFormatError,
        match="SKILL.md frontmatter YAML 解析失败",
    ) as exc_info:
        parse_frontmatter("description: [unterminated")

    assert isinstance(exc_info.value.__cause__, yaml.YAMLError)
    assert "reason" in exc_info.value.context


@pytest.mark.parametrize("frontmatter", ("- one\n- two\n", "plain scalar", "42"))
def test_non_mapping_frontmatter_is_rejected(frontmatter: str) -> None:
    with pytest.raises(
        IrisSkillFormatError,
        match="SKILL.md frontmatter 顶层必须是对象",
    ):
        parse_frontmatter(frontmatter)


def test_unknown_frontmatter_fields_are_preserved() -> None:
    parsed = parse_frontmatter(
        "description: Example skill\n"
        "allowed-tools:\n"
        "  - read_file\n"
        "disable-model-invocation: true\n"
        "version: 2\n"
        "custom-key: custom value\n"
    )

    assert parsed == {
        "description": "Example skill",
        "allowed-tools": ["read_file"],
        "disable-model-invocation": True,
        "version": 2,
        "custom-key": "custom value",
    }
