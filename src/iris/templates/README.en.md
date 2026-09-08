[中文](README.md)

# `iris.templates`

`iris.templates` copies packaged agent templates into a caller-selected directory. The only public
API is `scaffold_template()` and the only current built-in template is `file-agent`. The CLI
currently provides only `iris chat`; scaffolding is a Python SDK operation.

## Quick start

```python
from iris.templates import scaffold_template

written = scaffold_template("file-agent", "./my-agent")
```

The template creates `agent.yaml`, `README.md`, and `README.en.md`. The generated YAML can be loaded
with `iris.agents.load_agent_config()`.

`scaffold_template(template_name, target_dir, *, overwrite=False)` returns the written target paths.
An unknown template raises `IrisTemplateNotFoundError`; an existing target file raises
`IrisTemplateError` unless `overwrite=True`. Conflict detection happens before copying.

`file-agent` selects OpenAI `gpt-4o-mini`, a basic system prompt, the read-only `file.read`,
`file.list`, and `file.grep` tools, `writes: confirm`, and `session.backend: none`. The template
contains configuration and documentation only; it does not implement an agent loop.

## Packaging and maintenance

The project uses the `uv_build` backend declared in `pyproject.toml`. Templates live inside the
`iris` package under `src/iris/templates/builtin/` and are distributed with it. When changing a
template, inspect the built artifact and verify that the scaffolded config loads and its tools
register successfully.

There are currently no dedicated template tests under `tests/`. Add scaffold behavior coverage
when adding or changing a template.

```bash
uv run ruff check src/iris/templates
```
