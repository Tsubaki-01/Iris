[中文](README.md)

# `iris.tools`

`iris.tools` is Iris's tool kernel. It adapts Python callables or `BaseTool` subclasses into
model-visible schemas and centralizes input validation, permission checks, execution, result
normalization, large-output artifacts, middleware, and circuit breaking.

## Architecture

```mermaid
flowchart TD
    Source["callable / BaseTool"] --> Definition["ToolDefinition + input_schema"]
    Definition --> Registry["ToolRegistry / ToolRegistryView"]
    Registry --> Executor["ToolExecutor"]
    Executor --> Permission["PermissionPolicy"]
    Executor --> Middleware["ToolMiddleware"]
    Executor --> Breaker["CircuitBreaker"]
    Executor --> Artifact["ToolArtifactStore"]
    Executor --> Result["ToolResult"]
```

## Quick start

```python
from pathlib import Path

from iris.message import ToolUseBlock
from iris.tools import ToolExecutionContext, ToolExecutor, ToolRegistry, tool


registry = ToolRegistry()


@tool(registry=registry, description="Create a greeting")
def greet(name: str) -> str:
    return f"Hello, {name}"


executor = ToolExecutor(registry)

result = await executor.execute_one(
    ToolUseBlock(id="call_1", name="greet", input={"name": "Iris"}),
    ToolExecutionContext(workspace_root=Path(".")),
)
```

## Definitions, registry, and schemas

`ToolDefinition` holds the validated name, description, object JSON schema, capabilities, group,
aliases, deferred flag, output limits, and metadata. `ToolExecutionContext` carries call, workspace,
session, agent, permission, metadata, shared read-state information, and a shared live
`cancellation` signal that serialization excludes. `ToolResult` is the single result boundary;
`model_content` produces model-facing text and `to_block_metadata()` keeps the supported metadata
subset.

`BaseTool` defines `validate_input()`, read/destructive/concurrency classification, and async
`arun()`. `CallableTool` derives a schema from signatures, annotations, docstrings, or an explicit
Pydantic model and normalizes strings, `None`, JSON-compatible values, and exceptions. Synchronous
functions use `CallableExecutionMode.INLINE` by default, preserving the existing calling thread and
ordering. Only an explicit `THREAD` declaration runs the function in a worker thread. Async
functions cannot use `THREAD` and fail registration with `IrisToolValidationError`. If a synchronous
thread function returns an awaitable, Iris still awaits it on the event loop. Preset kwargs are
hidden from schema and callers cannot override them.

`ToolRegistry` registers tools/functions, resolves names and aliases, creates filtered views, exports
active schemas, and searches deferred definitions. Deny filters override allow filters. Deferred
tools are hidden unless explicitly allowed. Schema helpers support Iris-native, OpenAI Chat,
OpenAI Responses, and Anthropic wrapper shapes; runtime's active provider path currently mounts the
OpenAI Chat shape.

`@tool` attaches metadata without wrapping the function. Passing `registry` immediately calls that
registry's `register_function()`; omitting it leaves registration to config assembly or a later
explicit call. Schema extraction supports the documented Python/Pydantic types and Google-style
docstring argument descriptions. Unsupported parameter types produce validation errors.

## Execution and HITL preflight

`execute_one()` always returns `ToolResult`, mapping not-found, validation, permission, execution,
middleware, and open-circuit failures to stable error codes. `execute_many()` runs consecutive
read-only concurrency-safe calls concurrently and serializes writes or unsafe calls while preserving
result order and shared file read state. Classification failure conservatively falls back to serial.

`prepare_many()` performs registry lookup, validation, and policy checks without running middleware,
the breaker, artifact persistence, or tool side effects. It returns `PreparedToolCall` objects and a
human request where required. `execute_prepared()` begins a new stage, revalidates current state,
accepts an approval only for the exact tool-call ID, and optionally accepts a `ToolEffectGuard`.
Historical approval never overrides current deny, schema, workspace, or stale-read checks.

`PermissionPolicy.fingerprint_payload()` is the lifecycle resumability contract for permission
state. It must return deterministic JSON-safe data; custom policies must implement it explicitly
rather than relying on object representations. `DefaultPermissionPolicy` includes its policy type,
payload version, and `write_mode`.

Preflight precedence is: deny returns `PERMISSION_ERROR`; a human tool under allow creates its own
question; a human tool under require-human fails closed to prevent nested gates; only an ordinary
require-human call creates a permission prompt.

After approval, execution checks the circuit breaker, cancellation, the effect guard, and
cancellation again before entering middleware `before_call`, tool `arun`, artifact handling,
middleware after hooks, and breaker accounting. A guard failure starts no tool effect. Cancellation
after a claim propagates as control flow to runtime instead of becoming a normal tool error.
Low-level executor callers may omit the guard; lifecycle execution requires it through
`ToolBridge`. Parallel context copies preserve identity for both shared read state and cancellation.
Cooperative cancellation uses `IrisCancellationRequestedError` from `iris.exceptions`, and
`CallableTool` propagates it instead of normalizing it as an ordinary tool error. A thread worker
cannot be forcibly terminated: cancellation or timeout stops waiting, abandons its late return, and
lets lifecycle settlement fail closed when a durable claim already exists.

The blocking I/O in `read_file`, `list_files`, and `grep_search` runs in worker threads; `write_file`
and `edit_file` remain inline. Workers never mutate shared `ReadFileState`. A read returns an
immutable `ReadFileRecord` observation that the event loop merges only after a successful await.

`ToolExecutor` provides classification, revalidation, and per-call execution primitives only. The
lifecycle active path layers a fixed internal runtime window bound of 8 over those primitives.
Only consecutive read-only and concurrency-safe calls can enter a window; STOP, HITL, preflight
results, and unsafe calls remain barriers. Every call keeps its own durable claim. Body completion
does not determine result order, and claim telemetry order is not an ordinal contract. Undeclared
synchronous callables remain inline and may block the event loop; explicit `THREAD` placement
isolates blocking waits but does not promise CPU speedup. Placement does not enter provider schemas.
Future
NETWORK/MCP or write concurrency must define a new effect and recovery protocol rather than merely
changing the capability classifier.

## Built-in file tools

`register_file_tools()` registers, in stable order, `read_file`, `list_files`, `grep_search`,
`write_file`, and `edit_file`, injecting one shared `WorkspaceFileService`.

```mermaid
flowchart LR
    Executor["ToolExecutor"] --> Adapter["FileTool.arun"]
    Adapter --> Hook["ConcreteTool._impl"]
    Hook --> Service["WorkspaceFileService"]
    Service --> Boundary["WorkspacePolicy / ReadFileState / filesystem"]
```

- reads may include `L0001 |` line numbers and update `ReadFileState` through loop-side observation
  merge;
- list uses streaming `os.scandir` discovery order and does not guarantee global lexicographic
  order; list patterns retain `Path.rglob()` recursion semantics, including `**` matching zero or
  more directory segments; grep reads UTF-8 files line by line and skips `.iris` before descent;
- list/grep stop as soon as the global `max_results` limit is reached, and `max_results=0` performs
  no path resolution, walk, stat, or open;
- recursive file discovery skips escaping symlinks and grep skips decoding failures;
- overwriting/editing an existing file requires a prior unchanged read;
- edit requires exactly one match;
- resolved parent/symlink escapes are rejected;
- successful paths use workspace-relative `/` separators.

Large non-error output is stored at `.iris/tool-results/{session_id}/{call_id}.txt` with sanitized
identifiers, and the result becomes a preview plus artifact metadata. Default permissions do not
directly allow writes; configure `DefaultPermissionPolicy(write_mode="allow")` or let runtime host
the confirmation gate.

## Human tool, middleware, breaker, and discovery

YAML name `human.ask` registers model-visible `ask_question`. `AskQuestionTool` converts validated
input to `QuestionPrompt` and refuses direct `arun()`; runtime owns the interaction.

`ToolMiddleware` supports before, after, error, and legacy `after_execute` hooks. Middleware failures
become `MIDDLEWARE_ERROR`. `CircuitBreaker` tracks consecutive failures by tool name and returns
`CIRCUIT_OPEN` during cooldown.

`DeferredToolIndex` uses a local BM25-like ranker over name, tags, group, and description, with CJK
bigrams, low-weight single characters, query coverage, and stable sorting. `ToolSearchTool` exposes
`tool_search` and returns matching definitions without activating them.

## Public surface and boundaries

The exact top-level API is `src/iris/tools/__init__.py::__all__`, including
`CallableExecutionMode`, and covering models, base/adapters,
registry/view, executor/preflight, permission/artifact/middleware/breaker types, file/human tools,
deferred discovery, schema helpers, and `tool`. Protected `_impl()` hooks and executor private
lifecycle methods are internal.

The package does not run provider loops, persist ordinary session data, render host UI, or provide
MCP implementations merely because `ToolCapability.MCP` exists.

## Maintenance

| Change | Main location | Tests |
| --- | --- | --- |
| Models, callable/schema adaptation, and registration | `base.py`, `schema.py`, `registry.py` | `tests/tools/test_registry.py`, `tests/tools/test_executor.py` |
| Lifecycle and HITL preflight | `executor.py`, `permissions.py` | `tests/tools/test_executor.py`, `tests/tools/test_executor_preflight.py`, `tests/tools/test_human_ask_tool.py` |
| File tools, artifacts, and workspace safety | `builtin/file.py`, `artifacts.py` | `tests/tools/test_file_tools.py` |
| Circuit breaker | `circuit.py` | `tests/tools/test_circuit_breaker.py` |

Deferred search in `discovery.py` currently has no dedicated test file.

```bash
uv run pytest tests/tools
uv run ruff check src/iris/tools tests/tools
```
