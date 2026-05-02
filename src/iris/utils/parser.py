"""Parse text files that optionally contain YAML front matter.

This module focuses on read/parse behavior and the immutable parsed payload.

Example:
    >>> text = "---\\nname: A\\n---\\nBody"
    >>> parsed = parse_file(text, "a.md")
    >>> parsed.name
    'A'
    >>> parsed.description
    ''
    >>> parsed.content
    'Body'
    >>> parsed.raw
    '---\\nname: A\\n---\\nBody'
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------
class ParsedFile(BaseModel):
    """Immutable representation of a parsed file.

    Attributes:
        path (Path): Original file path associated with the parsed text.
        metadata (dict): Metadata extracted from YAML front matter.
        content (str): Body text after front matter is removed.
        raw (str): Original, unmodified file content.

    Example:
        parsed = ParsedFile(path=Path("note.md"), metadata={}, content="...", raw="...")
    """

    model_config = {"frozen": True}

    path: Path = Field(description="The path to the file that was parsed.")
    metadata: dict = Field(
        default_factory=dict,
        description="The metadata extracted from the file.",
    )
    content: str = Field("", description="The content of the file.")
    raw: str = Field("", description="The raw content of the file.")

    @computed_field
    @property
    def name(self) -> str:
        """Return display name from metadata, falling back to filename stem.

        Returns:
            str: Metadata-provided name when available; otherwise `path.stem`.
        """

        return self.metadata.get("name", self.path.stem)

    @computed_field
    @property
    def description(self) -> str:
        """Return normalized description text derived from metadata.

        Returns:
            str: Trimmed description string. Non-string values are converted to
                strings to keep caller behavior consistent.
        """

        desc = self.metadata.get("description", "")
        return desc.strip() if isinstance(desc, str) else str(desc).strip()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
_FRONT_MATTER_RE = re.compile(
    r"\A[ \t]*---[ \t]*\n(?P<yaml>.*?)^[ \t]*---[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


def parse_file(text: str, path: Path | str | None = None) -> ParsedFile:
    """Parse text into metadata and content using YAML front matter.

    Front matter is recognized only when the document starts with a fenced YAML
    block. If parsing fails, the function raises a `ValueError` to provide
    caller-visible context about the file that failed.

    Args:
        text (str): Full file content to parse.
        path (Path | str | None): Source file path used for diagnostics and fallback naming.

    Returns:
        ParsedFile: Immutable parsed payload containing path, metadata, body
            content, and raw text.

    Raises:
        ValueError: If a front-matter block is detected but contains invalid YAML.

    Example:
        >>> parse_file("---\nname: A\n---\nBody", Path("a.md")).name
        'A'
    """

    metadata: dict = {}
    content = text

    # Only treat front matter as valid when it appears at the file start.
    match = _FRONT_MATTER_RE.match(text)
    if match:
        yaml_str = match.group("yaml")
        try:
            parsed = yaml.safe_load(yaml_str)
            if isinstance(parsed, dict):
                metadata = parsed
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML front matter in {path}: {e}") from e
        content = text[match.end() :].strip()

    return ParsedFile(
        path=Path(path) if path is not None else None,
        metadata=metadata,
        content=content,
        raw=text,
    )
