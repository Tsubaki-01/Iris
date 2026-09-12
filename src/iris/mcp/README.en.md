[中文](README.md)

# iris.mcp

Imports external MCP configuration and resolves its environment. This stage provides independent
configuration APIs; Agent YAML integration and runtime connections follow in later stages.

## Configuration APIs

Requires Python 3.12+; dependencies are installed with Iris. `config.load_mcp_config()` accepts
JSON/JSONC `mcpServers`, `servers`, or `mcp_servers`, and TOML `[mcp_servers.<name>]`.
It imports the MCP section from one explicit file and preserves original server names.

```python
import os
from pathlib import Path

from iris.mcp.config import load_mcp_config, resolve_server_config

config = load_mcp_config(Path("mcp.json"), overrides={})
servers = [
    resolve_server_config(server, environ=os.environ, workspace_root=Path.cwd())
    for server in config.servers
]
```

`load_mcp_config()` reads and validates declarations without connecting. Disabled entries skip
all remaining fields. Servers are required by default: invalid required declarations raise
`IrisConfigError`; optional failures appear in `config.diagnostics`.
`agents.config.mcp.MCPServerOverride` accepts required, trust_annotations, and the two timeouts.
`AgentMCPConfig` declares the file reference and is not yet part of `AgentConfig`.

`resolve_server_config()` expands `${VAR}`, `${VAR:-default}`, and `${env:VAR}` exactly once.
STDIO environment precedence is env_vars → envFile → env; the host environment stays unchanged.
Relative cwd/envFile paths use the MCP file directory; omitted cwd uses workspace_root.
HTTP headers are merged by lowercase name after expansion; conflicting effective values fail.
HTTP aliases and explicit SSE normalize into one internal transport field; URLs never imply SSE.

## Implementation and maintenance

- `config.py`: the single owner of source normalization and environment resolution.
- `models.py`: external declarations and internal config/resolved/diagnostic data.
- `../agents/config/mcp.py`: Iris file reference and policy models.
- `tests/mcp/test_config.py`, `test_environment.py`: copied configuration, disabling, conflicts,
  and environment precedence.

Set `UV_CACHE_DIR` at the repository root, then use `uv run pytest`, `uv run ruff check`, and
`uv run mypy`, targeting `tests/mcp`. Client inputs, command substitutions, plugin variables,
remote executors, dynamic header helpers, and OAuth UI are outside the first release. Enabled
declarations containing these fields or expressions fail explicitly.
