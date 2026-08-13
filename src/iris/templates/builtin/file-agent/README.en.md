[中文](README.md)

# `file-agent`

`file-agent` is Iris's minimal local file-assistant template. It contains `agent.yaml`, `README.md`,
and `README.en.md`.

The default config selects OpenAI `gpt-4o-mini`, uses a simple system prompt, exposes only
`file.read`, `file.list`, and `file.grep`, keeps `writes: confirm`, and disables SQLite sessions.
It does not enable write tools and does not implement an agent loop.

```python
from iris.agents import build_tool_registry, load_agent_config

config = load_agent_config("agent.yaml")
registry = build_tool_registry(config.tools)
```

Change `model.provider` and `model.name` to choose another model. Add `file.write` or `file.edit`
only when writes are required, and set `session.backend: sqlite` when durable history and HITL
recovery are required.

There are currently no dedicated template tests under `tests/`. Add scaffold behavior coverage
when changing this template.
