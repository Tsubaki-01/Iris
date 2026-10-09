[中文](README.md)

# `file-agent`

`file-agent` is Iris's minimal local file-assistant template. It contains `agent.yaml`, `README.md`,
and `README.en.md`.

The default config selects OpenAI `gpt-4o-mini`, uses a simple system prompt, exposes only
`file.read`, `file.list`, and `file.grep`, keeps `writes: confirm`, and disables SQLite sessions.
It does not enable write tools and does not implement an agent loop.
The default protocol is `responses`. `permissions.workspace: .` resolves relative to this `agent.yaml`.
`session.backend: none` keeps session state in process memory and loses it on exit.

In a Python environment with Iris installed, enter this directory, configure
`IRIS_PROVIDER_API_KEYS__OPENAI`, and start:

```bash
iris chat agent.yaml
```

If credentials are stored in this directory's `.env`, explicitly use
`iris chat agent.yaml --env-file .env`. The following SDK fragment only loads configuration and builds
the tool registry; it does not call a model:

```python
from iris.agents import build_tool_registry, load_agent_config

config = load_agent_config("agent.yaml")
registry = build_tool_registry(config.tools)
```

Use the CLI or `iris.harness.AgentRunner.from_config_path("agent.yaml")` for complete execution.
SDK hosts must `await runner.aclose()` when finished.

Change `model.provider` and `model.name` to choose another model. Add `file.write` or `file.edit`
only when writes are required, and set `session.backend: sqlite` when durable history and HITL
recovery are required.

There are currently no dedicated template tests under `tests/`. Add scaffold behavior coverage
when changing this template.
