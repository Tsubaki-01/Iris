[中文](README.md)

# Local MCP example

Run from the Iris repository root with Python 3.12+ and the synchronized uv environment.
The server uses the official MCP SDK included with Iris and provides read-only echo and local
time queries. The server itself requires no external API or credentials.

| Tool | Input | Result |
| --- | --- | --- |
| `echo` | `text` | The original text |
| `get_current_time` | `timezone_name`, default `Asia/Shanghai`; also accepts `UTC` | `timezone`, offset-aware `iso_time`, and integer `unix_timestamp` |

The time tool reads the host system clock with one-second precision. Beijing time uses its
current UTC+08:00 offset; historical conversions and other time zones are outside this example.
The tool schema rejects unsupported zones. Time results are preserved as MCP structured content
and an Iris JSON artifact.

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv sync
uv run iris chat examples/mcp/agent.yaml
```

Actual chat requires a DeepSeek key, `IRIS_PROVIDER_API_KEYS__DEEPSEEK`, or a different model
in `agent.yaml`. Ask the agent to echo some text through MCP. Iris starts the server before
the first execution, reuses its connection across conversations, and waits for ongoing calls
before closing the process on exit.

To query time, ask for the current Beijing and UTC times through MCP. The model calls
`mcp__local__get_current_time`; accuracy depends on the host system clock.

`mcp.json` and the Codex-format `mcp.toml` point to the same service. Change `mcp.path` in
`agent.yaml` to `mcp.toml` to switch formats. Both use `cwd: .`, relative to the MCP config file,
so Python runs `examples/mcp/server.py`. The `uv run` environment provides Python with Iris
dependencies. Other hosts must provide the same runtime on PATH or configure its absolute path.
A missing executable makes required-server preparation fail before a run is created.

Local read-only trust is declared in `agent.yaml` overrides. Automatic permission requires both
local `trust_annotations` and the server's `readOnlyHint`; other tools follow ordinary HITL.

Verify real startup, calls, shutdown, and time values without calling an LLM:

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run pytest -p no:cacheprovider --basetemp="$PWD\tmp\pytest-mcp-example-check" tests/examples/test_mcp_examples.py tests/examples/test_mcp_time.py
```

Choose a fresh basetemp directory for each manual run. See the [MCP package](../../src/iris/mcp/README.en.md)
for configuration, result artifacts, and limitations, and the [harness guide](../../src/iris/harness/README.en.md)
for SDK lifecycle responsibilities.
