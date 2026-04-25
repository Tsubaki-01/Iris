"""Build text files with optional YAML front matter."""

import yaml


def build_file(content: str, metadata: dict | None = None) -> str:
    """Construct file text with optional YAML front matter.

    Args:
        content (str): Body text to include after the front matter.
        metadata (dict | None): Optional metadata to serialize as YAML front
            matter. If `None` or empty, no front matter will be included.

    Returns:
        str: Combined file text with YAML front matter if metadata is provided.

    Example:
        >>> build_file("Body", {"name": "A"})
        '---\\nname: A\\n---\\nBody'
    """

    if not metadata:
        return content

    yaml_str = yaml.safe_dump(metadata, sort_keys=False).strip()
    return f"---\n{yaml_str}\n---\n{content}"
