[中文](README.md)

# iris.mcp

Imports external MCP configuration and provides SDK connections, multi-server catalog publication,
and tool adapters, integrated with Agent YAML, root/child runners, and the CLI.

See [examples/mcp](../../../examples/mcp/README.md) for a runnable local service and JSON/TOML
configuration. Stable package exports are `MCPManager` and `load_mcp_config`; configuration models
are exported from `iris.agents`.

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
`AgentConfig.mcp` uses `AgentMCPConfig`; `mcp.path` resolves relative to the agent YAML.
Shared assembly reads declarations. Root `AgentRunner.aprepare()` prepares before execution and
publishes the complete catalog. Connections span runs; the host awaits every
original execution call fully before awaiting `runner.aclose()`.
Children use the same YAML configuration with independent connections, prepare before admission, and
close at WAITING/completion. Recovery rediscovers tools using current configuration. Child closure
does not affect root connections; the existing parent/child policy decides permissions.

`resolve_server_config()` expands `${VAR}`, `${VAR:-default}`, and `${env:VAR}` exactly once.
STDIO environment precedence is env_vars → envFile → env; the host environment stays unchanged.
Relative cwd/envFile paths use the MCP file directory; omitted cwd uses workspace_root.
HTTP headers are merged by lowercase name after expansion; conflicting effective values fail.
HTTP aliases and explicit SSE normalize into one internal transport field; URLs never imply SSE.

## Single-server connections

Construct `connection.MCPConnection(resolved)` synchronously, then await `open()` and
`list_tools()`. Call `call_tool(name, arguments)` with original wire names and finish with
`aclose()`. `protocol_version` reports the negotiated version. One long-lived task enters and
exits SDK contexts; callers await the session directly, serialized per server. The call timeout
includes lock waiting and up to eight state-only continuation rounds. Cancellation reaches the SDK.
SSE idle waiting does not spend the tool-call budget; preparation and calls retain outer deadlines.

The lockfile selects MCP SDK 2.2.0 and httpx2 2.12.0. STDIO, Streamable HTTP, and SSE use SDK
transports with auto negotiation. Real `AgentRunner → executor → SDK → server` tests cover:

| Transport | Protocol | Local fixture |
| --- | --- | --- |
| STDIO / Streamable HTTP | 2026-07-28 | SDK 2.2.0 Server, modern discover |
| STDIO / Streamable HTTP | 2025-11-25 | SDK handshake-only driver / legacy HTTP initialization |
| Explicit SSE | 2025-11-25 | SDK handshake-only driver |

The fixture is iris-test v1. A separate regression covers modern SSE idle reuse. HTTP tests verify
static/env headers and client closure; STDIO tests verify process exit. No public MCP service or paid
LLM invocation is required.

SDK MCPError becomes `IrisMCPCallError` without inferring whether the remote operation ran.
Known output/MRTR failures use `IrisMCPToolError.code`. Only the SDK validates successful output
schemas; 2.2.0 treats null as missing and reports `MCP_RESULT_INVALID` when a schema is declared.
Iris does not replay calls or reconnect automatically.

## Implementation and maintenance

`manager.MCPManager(config, registry=registry, workspace_root=workspace).prepare()` resolves,
connects, and fully discovers servers in server-id order. Each server shares one startup deadline.
All candidates publish through `registry.register_many()` once, becoming visible to existing
ToolRegistryViews. Required failures publish no partial MCP tools; optional failures produce
diagnostics. Valid empty catalogs are allowed.

The successful `snapshot` stays fixed, and concurrent/repeated prepare calls do not reconnect or
rediscover. Failed managers close; create a new instance for new configuration. `aclose()` attempts
every owned connection and reports explicit closure failures to the host. Effective env/header
values in the snapshot stay in memory and should not be logged or persisted.

`catalog.build_catalog(server, sdk_tools)` filters original wire names and compiles JSON Schema
2020-12 input validators. Local references work; unresolved external references and other dialects
produce diagnostics and exclude the tool. Public names use ASCII letters, digits and underscores
in `mcp__...`, adding a stable identity
digest only when necessary. Register `tools.MCPTool(descriptor, connection)` in the existing
ToolRegistry to use the ordinary input, permission, execution, and cancellation path.

Default permissions allow MCP tools only when local trust_annotations and readOnlyHint are both
true. Other tools require confirmation. Uncertain SDK failures become ordinary errors or
OUTCOME_UNKNOWN according to that policy; Iris interruption with an unsettled claim follows the
existing unknown path even for trusted reads. MCP tools remain outside parallel windows.
The existing executor cancels SDK requests and waits for cleanup before settlement; cancellation
does not prove remote effects stopped. CLI shutdown waits for `manager.close(cancel_run=True)`,
closes the runner, then finishes output and the background loop.

Rich content, structuredContent, SDK-retained metadata, and oversized projections are saved as
complete `.mcp.json` files. Models receive bounded text and a path. If the text after middleware
exceeds the limit, the JSON stays at `artifact.path` and a separate `.model.txt` file preserves the
complete final text at `artifact.text_path`. Large text without a native artifact uses one `.txt`
file for both fields. Raw MCP JSON and final model text are separate representations, not substitutes.
The adapter preserves full text for middleware; the executor alone applies the final limit using
ToolDefinition.preview_chars. Direct MCPTool.arun calls return results before this final limit.
Files are scoped to the current context.session_id and receive a fresh random identifier on each
write; error paths appear in the model-visible error.message.
Complete runners with the default context policy attach a historical result reference.
`context_read` retrieves final model text with `representation="text"` or native MCP JSON with
`"raw"`, without another remote call. See [context reads](../tools/README.en.md#current-session-context-reads)
for scope and pagination.
After the SDK returns, complete projection, `model_dump`, JSON encoding, and writing run in a tool IO
worker. Cancellation drains this local job and preserves the known result or `ARTIFACT_ERROR`
without another remote call. The SDK network wait remains cancellable.

- `config.py`: the single owner of source normalization and environment resolution.
- `models.py`: external declarations and internal config/resolved/diagnostic data.
- `connection.py`: SDK transport ownership, complete pagination, calls, and closure.
- `tests/mcp/test_interoperability.py`: five complete protocol paths and new-run reuse after cancellation.
- `catalog.py` / `tools.py`: descriptors and the existing BaseTool adapter.
- `manager.py`: prepare/close coordination and the single catalog snapshot.
- `../agents/config/mcp.py`: Iris file reference and policy models.
- `tests/mcp/test_config.py`, `test_environment.py`: copied configuration, disabling, conflicts,
  and environment precedence.
- `tests/mcp/test_connection.py`, `test_sdk_contract.py`: Iris scheduling and real SDK contracts.

Set `UV_CACHE_DIR` at the repository root, then use `uv run pytest`, `uv run ruff check`, and
`uv run mypy`, targeting `tests/mcp`. Client inputs, command substitutions, plugin variables,
remote executors, dynamic header helpers, and OAuth UI are outside the first release. Enabled
declarations containing these fields or expressions fail explicitly.
