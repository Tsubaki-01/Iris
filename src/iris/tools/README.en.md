[中文](README.md)

# `iris.tools`

`iris.tools` is Iris's tool kernel. It adapts Python callables or `BaseTool` subclasses into
model-visible tool definitions and centralizes input validation, permission checks, execution, result
normalization, large-output artifacts, middleware, and circuit breaking.

## Architecture

Internal `subagent.py` defines the fixed `subagent(prompt, agent?)` schema and immutable routes.
`agent` selects an exact catalog key, defaulting to the catalog default; `prompt` is trimmed and
must be nonblank. `SubagentTool.execute_subagent()` returns `ToolResult | ChildWaiting` through
a narrow port. Direct `arun()` raises `IrisToolExecutionError`; ordinary `BaseTool.arun()` retains
its terminal `ToolResult` contract.

The default policy allows only the concrete builtin `SubagentTool`; other AGENT tools still need
human approval. Internal `MostRestrictivePermissionPolicy` evaluates both policies against each
actual child tool and selects the original decision by DENY > REQUIRE_HUMAN > ALLOW. Ties retain
the parent's reason/metadata.

The dedicated `ToolExecutor` Sub Agent path preserves raw input parsing, fresh permission refresh,
and final identity/artifact normalization. Linked continuation skips outer permission. ChildWaiting
returns directly, without middleware, breaker, parent claim, or ordinary timeout. Ordinary tools
normalize and persist the final result after the middleware chain returns. Controller lifecycle, persistence,
and recovery errors propagate unchanged. ACTIVE and WAITING share final normalization and
`ARTIFACT_ERROR` projection.

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

## Importing tool images

Ordinary tools can call `import_tool_image(source: Path | bytes, context, *, name=None) -> ImageBlock`
and place the returned block in `ToolResult.content`, interleaved with `TextBlock` values.
[`images.py`](images.py) takes the workspace/session from `ToolExecutionContext`, resolves relative
paths against that workspace, and reuses `utils.images.save_image()` to save images under
`.iris/image-cache/<encoded session>/`. The helper performs synchronous image I/O and processing;
async tools should call it within their existing I/O worker operation.

Ordinary files and bytes create a new snapshot on each import, unaffected by later source changes.
Files already within the workspace's image-cache are still decoded and processed against the
dimension/size policy. Compliant files serve as both original and model references; when a transform
is needed, the original stays in place and only a new model copy is written under the current
session. Cache membership never bypasses image processing. Blocks are returned only after saving
completes. Read, decode, processing, or save failures raise `IrisImageError` through the existing tool
failure path. Tools produce references; they do not choose API protocols, encode base64, or invoke
an auxiliary vision model.

## Explicit command execution

Declare `exec.command` in Agent `tools.builtin` to expose `exec_command`. Its only parameters are
`command`, `cwd` (default `.`), and an optional positive `timeout_seconds`. Root `command`
configuration selects the environment; the model cannot change the mode, image, or mount. See
[agents](../agents/README.en.md#command-environment). SDK callers can construct
`ExecCommandTool(CommandBinding(config, service, environment))`; the host owns service preparation
and closure.

Declare `exec.python` to expose `run_python(code, cwd='.', timeout_seconds=None)`, or register
`RunPythonTool` through the SDK. Both command tools share the root CommandBinding, EXECUTE policy,
and stopping rules. Code must contain non-whitespace text. Native uses Iris's `sys.executable`;
Docker uses Python from the selected image, with dependencies prepared by the developer. Each call
starts a fresh process and passes source through a temporary file without shell quoting. Variables
do not persist; workspace files do. Use `print` for text output. Tracebacks go to stderr for the next
model turn. Imports resolve from cwd, and tracebacks retain source lines under `<iris-python>`.
There is no real script-path contract or persistent kernel. Use `exec_command` for existing scripts
or another interpreter.

`WorkspacePolicy` resolves cwd once within the current Agent workspace. Native commands run with
host-user access; cwd is not a filesystem boundary. Docker root and child calls share the root
`/workspace` mount. A narrower child workspace sets its default cwd and native file-tool scope;
child `writes: deny` does not make Docker commands read-only. The root mount controls their write
access. Each command uses a fresh shell without persistent `cd/export`, interactive stdin, or TTY.

`DefaultPermissionPolicy(execute_mode="confirm"|"allow"|"deny")` handles EXECUTE independently,
defaulting to confirmation. Commands retain preflight, HITL, permission refresh, effect claim,
and the existing non-read-only serial barrier. `BaseTool.timeout_owner` defaults to
`ToolTimeoutOwner.RUNTIME`; both command tools use `TOOL` and take the minimum of the configured, requested,
and context `tool_timeout_seconds` limits. The outer run owner handles the run deadline.

Exit zero succeeds; a nonzero exit, including 124/137, becomes `COMMAND_FAILED`. Local timeout is
`COMMAND_TIMEOUT`; confirmed cancellation and shared-stop interruption are `COMMAND_CANCELLED`
and `COMMAND_ENVIRONMENT_INTERRUPTED`. A known unstarted command is `COMMAND_UNAVAILABLE`.
Error messages contain exit status and diagnostics. Unknown execution propagates
`IrisToolOutcomeUnknownError` without replaying the command.

Both backends retain bounded output heads and tails, and the final model preview also keeps both
ends. Status and collection details precede stdout, with stderr last. Result `data` holds mode,
status, exit code, cwd, duration, derived `output_truncated`, and an `output_stats` object containing
collected/retained byte counts and truncation reasons. It does not duplicate stdout/stderr. Artifacts
save the retained body after middleware; discarded raw log bytes cannot be retrieved.
Live receipts and cleanup failures use excluded `context.command_stop_slot`,
whose identity survives context copying and middleware result replacement. A known result with
failed cleanup is returned for normal commit while the slot retains `cleanup_error` for outer
settlement. Cleanup errors without known facts propagate directly, including through middleware.
See [command](../command/README.md) for backend behavior and stopping scope.

## Definitions, registry, and schemas

`ToolDefinition` holds the validated name, description, object JSON schema, capabilities, group,
aliases, deferred flag, output limits, `preview_mode`, `context_retention`, and metadata. `ToolExecutionContext` carries call, workspace,
session, agent, permission, metadata, shared read-state information, and a shared live
`cancellation` signal that serialization excludes. `ToolResult` is the single result boundary:
its `content` is an ordered `list[DataBlock]`, `model_blocks` produces the complete model body,
and `model_content` extracts only text from that same projection. `to_block_metadata()` keeps the
supported metadata subset. `to_msg()` preserves all projected blocks in the history message,
without normalizing metadata again. Runtime commits and terminal tool closure share this projection.

`DataBlock` is `TextBlock | ImageBlock`, defined in [`iris.message`](../message/README.en.md).
Successful results and errors without an `error` object retain their original block order.
When `is_error=True` and `error` exists, `model_blocks` replaces the text with one authoritative
`Error[code]: message` block, followed by images in their original order. The original `content`
is unchanged. Neither the text-only `model_content` nor `artifact`/`data` represents the full
image body sent to a model.

`ToolDefinition.context_retention` defaults to `"keep"`. Authors can explicitly choose
`"observation"` to let runtime shorten committed successful result bodies under request pressure,
while retaining the original through `context_read`. Built-in `read_file`, `list_files`, `grep_search`,
`web_search`, and `web_fetch` opt in. Write/edit tools, command execution, subagent final answers,
HITL, Skill bodies, and unknown custom tools keep the default. A READ capability alone does not
make a result eligible.

After execution, executor finalization stamps `context_retention` and canonical `context_tool_name`,
overriding same-named metadata supplied by the tool. History stores these facts in
`ToolResultBlock.metadata.extra`; recovery uses the saved declaration and name rather than the
current registry. Errors and unclosed calls are never reduced.

Retention changes only the model's history view. Identical arguments still cause a real execution;
only afterward can equal canonical names, normalized JSON arguments, and complete result bodies
allow older duplicates to be folded. Many distinct medium-sized results can instead be shortened
without any individual result reaching the artifact threshold. See
[runtime](../runtime/README.en.md#history-projection-and-summary-construction) for group protection,
budget checks, and read-availability requirements.

`BaseTool` defines `validate_input()`, read/destructive/concurrency classification, and async
`arun()`. `CallableTool` derives a schema from signatures, annotations, docstrings, or an explicit
Pydantic model and normalizes strings, `None`, JSON-compatible values, and exceptions. Synchronous
functions use `CallableExecutionMode.INLINE` by default, preserving the existing calling thread and
ordering. Only an explicit `THREAD` declaration runs the function in a worker thread. Async
functions cannot use `THREAD` and fail registration with `IrisToolValidationError`. If a synchronous
thread function returns an awaitable, Iris still awaits it on the event loop. Preset kwargs are
hidden from schema and callers cannot override them.

`THREAD` only selects execution placement through `asyncio.to_thread()`; `CallableTool.arun()`
does not independently consume `context.cancellation`. Direct `arun()` callers own cancellation
of their task. Through the executor, the signal cancels the current wrapped call and waits for
downstream work to drain.

Each `CallableTool` uses one input model. Without an explicit `input_model`, Iris builds a
Pydantic model from the function annotations and docstring parameter descriptions. That model
owns both exported JSON Schema and first input validation, including fixed tuple positions,
`Annotated` constraints, nullable fields, and defaults. The callable receives validated fields
directly, preserving Python types such as tuples and nested `BaseModel` instances.

`schema_from_callable(func, preset_kwargs=...)` exports through the same dynamic model builder.
It requires resolvable annotations, supports ordinary and keyword-only parameters, and skips
`*args`/`**kwargs`. `schema_from_pydantic_model(model)` returns the complete model JSON Schema,
including `$defs` and root constraints such as `additionalProperties`.

`ToolRegistry` registers tools/functions, resolves names and aliases, creates filtered views, exports
logical definitions through `active_specs()`, and searches deferred definitions. Deny filters override
allow filters. Deferred tools are hidden unless explicitly allowed. `ToolSpec` projects validated
`name`, `description`, and `input_schema` fields with `strict=False`; execution policies stay in the
registry. [Provider adapters](../providers/README.en.md) own Responses and Chat Completions wrappers.
`ToolRegistryView.available_tools` includes deferred definitions within the same static filters;
only the host's original `allow` can bypass a group filter. `specs_for(names)` exports complete
logical definitions for selected canonical names in registration order without changing the shared view.
`search_deferred(query, include_groups=None, limit=10, allowed_names=None)` filters before ranking
and applying the limit.

Name-conflict checks use the registry's existing name and alias indexes directly instead of
revisiting every registered tool definition.

`@tool` attaches metadata without wrapping the function. Passing `registry` immediately calls that
registry's `register_function()`; omitting it leaves registration to config assembly or a later
explicit call. Schema extraction supports the documented Python/Pydantic types and Google-style
docstring argument descriptions. Unsupported parameter types produce validation errors.

## Execution and HITL preflight

`execute_one()` returns `ToolResult` for ordinary failures, mapping not-found, validation, permission, execution,
middleware, and open-circuit failures to stable error codes. `execute_many()` runs consecutive
read-only concurrency-safe calls concurrently and serializes writes or unsafe calls while preserving
result order and shared file read state. Classification failure conservatively falls back to serial.
Cancellation, unknown execution, and incomplete-cleanup control exceptions propagate.

`register(tool)` adds a `BaseTool` and raises a validation error on name or alias conflicts.
`prepare_many()` performs registry lookup, input-schema validation, and the initial policy check
without running middleware, the breaker, artifact persistence, or tool side effects. Each
`PreparedToolCall` retains both typed `validated_input` and normalized `arguments`. Runtime reuses
that plan for the complete tool batch. `execute_prepared()` refreshes permission only; it does not
repeat registry lookup, schema validation, or a typed-model-to-dict round trip. It accepts approval
only for the exact tool-call ID and optionally accepts a `ToolEffectGuard`. Historical approval
never overrides current deny, workspace, or stale-read checks.

Custom permission policies implement `PermissionPolicy.check(tool, params, context)`.

Preflight precedence is: deny returns `PERMISSION_ERROR`; a human tool under allow creates its own
question; a human tool under require-human fails closed to prevent nested gates; only an ordinary
require-human call creates a permission prompt.

After approval, execution checks the circuit breaker, cancellation, the effect guard, and
cancellation again before entering the `wrap_tool_call` chain, a pre-body cancellation check, tool
`arun`, the returning wrappers, and final identity/artifact handling. The breaker records the actual
body result once; short circuits, middleware post failures, and artifact errors do not count as
body failures. A guard failure starts
no tool effect. Cancellation after a claim propagates as control flow to runtime instead of becoming
a normal tool error.
Low-level executor callers may omit the guard; lifecycle execution requires it through
`ToolBridge`. Parallel context copies deep-copy only the isolated `metadata`; typed
`ReadFileState` and cancellation are shared directly without copying their objects or records.
Each new call gets independent command/control slots. Context projections within the same call
preserve slot identity; a subsequent call does not inherit the previous call's stop state.
`ToolExecutionContext` parses read state at its public raw-input boundary; file
services consume the typed object directly without repeating type checks.

The executor performs artifact handling once after the middleware chain returns. `call_next()`
returns the full downstream result; expanded final content is also subject to `max_result_chars`.
Cooperative cancellation uses `IrisCancellationRequestedError` from `iris.exceptions`, and
`CallableTool` propagates it instead of normalizing it as an ordinary tool error.

`ToolExecutor` monitors the whole call, including middleware and `arun()`, for async callables,
custom async `BaseTool` implementations, and THREAD callables. The body checks the signal again
before starting. A signal, external `Task.cancel()`, timeout, or runtime sibling cancellation sends
the first cancellation to the current call and waits for downstream work to drain; repeated requests
do not interrupt cleanup again. A completed body or a normal return after catching `CancelledError`
remains a known `ToolResult`. Runtime commits it in order before propagating cancellation or settling
the timeout. A known result does not clear the interruption; a genuinely cancelled body or unknown
exception keeps the existing control path.

A pending request before a wrapper enters its continuation prevents body startup. Cancellation or
cleanup failure after a known body result does not replay the tool: with a Runtime owner, the executor
returns the known result and deferred control so Runtime commits in order before settlement.
Low-level executor calls without that owner propagate control exceptions to their caller. Finite
artifact I/O still drains to recover its actual result.
Slow middleware, coroutines that suppress `CancelledError`, and INLINE blocking can still delay
exit. Cancellation of a custom THREAD callable ends only its async waiter; the worker may continue. Unresolved claims
still settle as `OUTCOME_UNKNOWN`, including read-only calls, and late returns cannot change the
durable result.

The blocking I/O in `read_file`, `list_files`, `grep_search`, `write_file`, and `edit_file` runs in
worker threads. Writes and edits use a snapshot of the existing immutable read records, complete
the checks, write, and stat in one job, then return a `ReadFileRecord` for loop-side merge. Workers
never mutate shared `ReadFileState`; a failed operation does not merge or replace other records.
The synchronous `WorkspaceFileService` methods remain available for direct callers.

Final result normalization, serialization, and artifact writing also run in a worker. These finite
local operations drain through repeated cancellation to recover their actual result or error. They
use the existing thread pool, with no new persistent worker or registry. This does not change direct
`CallableTool.arun()` THREAD cancellation or make remote requests non-cancellable.

`WorkspaceFileService.read_text_observed()` supplies complete text and a file observation from one
open file for Skill loading, sharing the workspace and regular-file boundaries.
It does not update shared read state; callers merge after a successful await. Ordinary
`read_file_observed(..., max_chars=...)` first examines a short header to distinguish images from text.
For text it reads only the budgeted page plus one lookahead character; preceding lines and columns
are skipped in chunks without loading a whole long line. Images return an `ImageBlock` with no read
observation. `read_text_observed()` remains text-only.

`ToolExecutor` provides classification, permission refresh, and per-call execution primitives only. The
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

When a memory service supplies a file view, Agent assembly binds the formal Markdown paths for
`read_namespaces` to `WorkspaceFileService(memory_view=...)`. Direct access and recursive traversal
inside that memory root use only the permitted namespaces' formal projections; legacy mixed files
and the database are outside this file view. Read and grep check the consumed source revision
against state observed after reading and include stale warnings, including grep with no matches.
Missing formal files still report `FILE_NOT_FOUND` with their projection status. Projected content
can be read; changes go through memory write tools or the SDK. Without a memory service, or with an
independent SDK service without a mirror, no memory file view is bound. When `memory.enabled` is on,
SQLite services constructed from configuration provide a mirror. Ordinary workspace files keep the existing rules.
The model's `memory_search` / `memory_fetch` tools read SQLite directly and do not depend on projections.

The Agent's memory switch automatically registers these two reads; memory writes remain explicit.
Standalone SDK `register_memory_tools()` still defaults to an empty selection; see
[memory](../memory/README.en.md).

`register_file_tools()` registers, in stable order, `read_file`, `list_files`, `grep_search`,
`write_file`, and `edit_file`, injecting one shared `WorkspaceFileService`.

`file.read`/`read_file` reads both text and static PNG/JPEG/WebP images, selecting by a short file
header rather than extension. Image paths pass through the shared file service, then follow the
[tool image import](#importing-tool-images) contract and return reference text plus an `ImageBlock`.
`offset`, `column`, `limit`, and `with_line_numbers` apply only to text. Images ignore these parameters,
return the complete model copy, and create neither text-page cursors nor edit-related `ReadFileState`.
There is no additional image tool or configuration alias.

Write/edit operations share a process-wide lock per resolved path, spanning cancellation and
freshness checks through replacement and final stat. Different service instances, sessions, roots,
and children serialize changes to the same file; a stale writer receives `STALE_FILE_STATE` after
acquiring the lock. Different paths can proceed concurrently. There is no FIFO promise or
coordination with shell commands, user Python code, external editors, or other processes.
The actual worker holds the lock, so cancelling its waiter cannot release it early. Business
cancellation while waiting produces `FILE_OPERATION_CANCELLED` before mutation; an operation
already in progress returns its actual result instead of reporting a completed write as unstarted.

A successful `edit_file` returns `data.file_change` with `file_path` and `patch`. Paths are relative
to the effective workspace with POSIX separators. The unified diff uses the actual old/new text,
normalized LF and no-final-newline markers. Model text stays a short summary; SDK and durable tool
results retain the patch. `write_file` does not return one. Synchronous
`WorkspaceFileService.edit_file()` still returns a string; the internal edit observation also
carries the read record and patch.

Explicit `file.publish` registration exposes `publish_artifact(file_path)`. SDK callers can register
`PublishArtifactTool(file_service=...)`; it is not added to `register_file_tools()` defaults.
The tool uses READ permission and the effective workspace boundary. It needs no prior `read_file`
and does not update edit/write read-state records.

Finish generating the file before publishing it. Publication copies the selected file in chunks to
a unique session artifact path, preserving its extension and reporting MIME, actual size, and a short
preview through `ToolResult.artifact`. Subsequent source edits/deletion or runner closure do not change
the copy. Failed copies remove their partial destination and return a tool error without retrying.
The tool performs no directory scan, cloud upload, or binary injection into model messages.

Hosts obtain the local path from the result or its `AgentRunner.list_tool_calls(run_id)` record.
SSE/WebSocket expose only artifact summaries and call identity; hosts provide their own display or
download interface. See the [Python report example](../../../examples/command/README.md).

```mermaid
flowchart LR
    Executor["ToolExecutor"] --> Adapter["FileTool.arun"]
    Adapter --> Hook["ConcreteTool._impl"]
    Hook --> Service["WorkspaceFileService"]
    Service --> Boundary["WorkspacePolicy / ReadFileState / filesystem"]
```

- text reads preserve decoded newlines, may include `L0001 |` line numbers, and update `ReadFileState`
  through loop-side observation merge;
- list uses streaming `os.scandir` discovery order and does not guarantee global lexicographic
  order; list patterns retain `Path.rglob()` recursion semantics, including `**` matching zero or
  more directory segments; grep reads UTF-8 files line by line and skips `.iris` before descent
  unless the explicitly requested path is inside `.iris`;
- list/grep stop as soon as the global `max_results` limit is reached, and `max_results=0` performs
  no path resolution, walk, stat, or open;
- with `max_results > 0`, a missing list/grep root raises `FILE_NOT_FOUND`;
- recursive file discovery skips escaping symlinks and grep skips decoding failures;
- overwriting/editing an existing file requires a prior unchanged read;
- edit requires exactly one match;
- resolved parent/symlink escapes are rejected;
- successful paths use workspace-relative `/` separators.

For text, `ReadFileInput` uses zero-based line `offset` and Unicode character `column` within that line.
`limit` defaults to 1000 lines and accepts 0..1000. The tool budgets source text, optional line
numbers, and a model-visible footer together:

```text
[read_file: offset=0, column=0; next_offset=0, next_column=800; has_more=true]
```

When has_more is true, pass next_offset/next_column as the next request's offset/column with the
same file path. False means EOF. Coordinates count source characters, excluding line-number
prefixes; consuming a newline advances offset and resets column. Long lines are split without
creating another artifact during ordinary paging. Page text retains trailing newlines. limit=0
only reports remaining content without consuming source text; no total-line scan is performed.
An invalid starting column returns COLUMN_OUT_OF_RANGE. An insufficient page budget returns
READ_BUDGET_TOO_SMALL. Direct service read_file/read_file_observed calls require max_chars.
The former returns `str | ImageBlock`; the latter also returns a `ReadFileRecord` for text or
`None` for images.

Large execution output, including errors, is stored at
`.iris/tool-results/{encoded_session_id}/{encoded_call_id}-{random_id}.txt`, and the result becomes a preview
plus artifact metadata. Each ID segment is `id_` followed by its complete UTF-8 bytes encoded as
lowercase hexadecimal; an empty ID becomes `id_`. Distinct IDs retain distinct paths even on
case-insensitive filesystems. Every write adds a random identifier, so reusing a call ID within a
session does not overwrite an earlier result. Containment is still checked when writing. Recovery
and forks reuse the saved immutable paths without copying or rewriting payloads.
Default permissions do not
directly allow writes; configure `DefaultPermissionPolicy(write_mode="allow")` or let runtime host
the confirmation gate.

`persist_json()` stores complete parsed MCP JSON; `artifact_store_for()` selects the store for
the current context.session_id. Oversized final `model_content` is saved after all middleware.
Plain text uses one `.txt` file, with `ToolArtifact.text_path == path`. When a native artifact already
exists, its `path` is preserved and a separate `.model.txt` file supplies `text_path`; the raw payload
cannot replace the text produced by middleware. Results within the limit need no additional text file.

`ToolResult.hook_feedback` is an empty tuple by default. Each feedback paragraph follows the original
body or `Error[code]: message`, marked with `[Hook feedback]`, inside the same `ToolResultBlock`.
It does not change the message role or overwrite the original `content`, `data`, or structured error.
This result projection is available independently; Agent YAML Hook wiring is provided separately.
Feedback shares the final text budget, and the saved full text contains every paragraph. Truncated
previews clear the structured `hook_feedback` after folding retained feedback into the preview, so
`model_content` and `to_msg()` do not append it again. Results within the limit retain the tuple.

`model_blocks` appends feedback text blocks after the complete body, including images.
Truncation preserves every image and its order; images do not consume the text character budget.
For mixed results, `text_path` starts with each image's name, MIME, original/model paths and dimensions,
then a blank line and the complete model text. Text-only files retain their existing layout. Image
references are not clipped with the preview, so `context_read` can locate either saved version.

Successes, returned errors, raised tool exceptions, and middleware errors share the final output
handling. The budget includes the error prefix and retrieval notice; `error.message` retains the
preview and path. If the notice and prefix alone exceed the budget, finalization returns
`ARTIFACT_ERROR` rather than emitting an incomplete reference or exceeding the character limit.
Preflight errors are clipped without writing files.

`ToolDefinition.preview_mode` defaults to `head`; both command tools use `head_tail`.
`ToolArtifact.preview` and final model text share the preview algorithm. The character budget
includes the error prefix, complete retrieval notice, and omission marker.
`ToolDefinition.preview_chars` controls preview length; `ToolExecutor` no longer accepts
`artifact_preview_chars`. An artifact write failure returns an error without retrying the write.
`ToolArtifactStore.persist_file(tool_use_id, source, preview=...)` stores a copy of an already
resolved source file for explicit publication.
The [MCP adapter](../mcp/README.en.md) uses the ordinary executor and cancellation bridge.
Default permissions allow only locally trusted read-only MCP tools.
`IrisToolOutcomeUnknownError` bypasses both exception conversions for existing runtime settlement.
Ordinary tools and MCP share this unknown exception. Its optional `stop_receipt` separately holds
process-local stop evidence, outside the generic error context and model-visible result.

## Current-session context reads

With the default `context_policy.enabled: true`, `AgentRunner` registers `context_read` and
`context_search` automatically. Neither belongs in `tools.builtin`, and neither requires memory or
`file.read`. The tools delegate through [`ContextAccessPort`](context_access.py), using the current
`ToolExecutionContext.session_id`; the model supplies neither a session ID nor an arbitrary file path.

| Tool | Parameters | Result and scope |
| --- | --- | --- |
| `context_read` | `ref`; `offset=0`; `limit=4000` (1..8000); `representation="text"` or `"raw"` | A saved text page; `ToolResult.data` contains `ref/representation/offset/next_offset/has_more/content` |
| `context_search` | Nonblank `query`; `after=0`; `limit=10` (1..20) | Unicode casefold substring search over committed text, tool previews, and image names/references in the current session; returns `matches/next_after/has_more` |

`message:<index>` names an original message; `result:<message_index>:<block_index>` names a tool
result block. Both indices are zero-based and are not renumbered by summary projection. Result
`text` reads the complete model text before final truncation, using `artifact.text_path` when
present or the archived body otherwise. Result `raw` requires an artifact and reads the native file
as text, such as MCP JSON. A message's text representation includes its role, sender, and block
boundaries.
Use the default `text` to recall ordinary historical content. Inline results and message references
have no `raw` representation; the tool parameter description makes this distinction explicit.

Message and inline-result text preserve image positions and display each image's name, original/model
paths, actual MIME types, and dimensions. Image-only results remain readable. Offloaded `text_path`
files already contain image references, so reading pages does not add another reference prefix.
To view an image again, use an already registered `read_file` on its model path. The original path is
available to the host or existing code tools for further processing. References remain readable when
`file.read` is not configured; context tools do not register it automatically, and the host can also
resubmit an existing ImageBlock.

Read offsets and limits count Python Unicode characters. The page and short continuation header
share a 12,000-character tool limit, avoiding another offload during ordinary paging. After hooks
still apply; exact page reconstruction assumes middleware does not rewrite the returned body.
Unavailable files or invalid references return `CONTEXT_SOURCE_UNAVAILABLE`; an unavailable raw
representation returns `CONTEXT_REPRESENTATION_UNAVAILABLE`. Reads never rerun the source tool or
substitute current file contents for its saved result.

Search scans at most 200 messages per call, returning at most one match per message and a snippet
of at most 240 characters. It matches saved image names and reference text without opening images,
performing OCR, or scanning raw artifacts and offloaded full text. A page with no matches can still
have `has_more=true`; continue from `next_after`. Matches identify the owning message/result: expand
them with context_read, then use read_file when image pixels are needed. Each complete read or search
operation runs in one IO worker.

## Human tool, middleware, breaker, and discovery

`ToolRegistry.register_many(tools)` checks names and aliases against both the batch and existing
indexes before publishing. Conflicts leave the registry unchanged. `register(tool)` uses the same
admission path, and existing views see the published tools.

YAML name `human.ask` registers model-visible `ask_question`. `AskQuestionTool` converts validated
input to `QuestionPrompt` and refuses direct `arun()`; runtime owns the interaction.

`ToolMiddleware` is an abstract base class with one required method:
`async wrap_tool_call(call: ToolCall, call_next: ToolNext) -> ToolResult`.
Inject instances through `ToolExecutor(..., middleware=[...])`. The first registered wrapper is
outermost: A before → B before → body → B after → A after.

`ToolCall` is a frozen view with `tool_use_id`, `tool_name`, `arguments`, `agent_id`, `session_id`,
optional `run_id`/`activation_id`, and `workspace_root`. It exposes neither a writable execution
context nor a `BaseTool` instance. Arguments are an independent snapshot; changing it does not
change the tool's actual input.

`call_next()` takes no arguments and may be called at most once. Omitting it can return a cached or
other substitute result; permission and claim checks have already run. A saved continuation expires
when its wrapper returns, and any downstream work already started must be collected before the call
finishes. Downstream results are read-only; return a new `ToolResult` to change them:

```python
from iris.message import TextBlock
from iris.tools import ToolCall, ToolExecutor, ToolMiddleware, ToolNext, ToolResult


class LabelResult(ToolMiddleware):
    """在下游结果后添加来源说明。"""

    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
        """返回新结果，保留原对象。"""
        result = await call_next()
        return result.model_copy(
            update={"content": [*result.content, TextBlock(text=f"Source: {call.tool_name}")]}
        )


executor = ToolExecutor(registry, middleware=[LabelResult()])
```

A wrapper can recover ordinary downstream exceptions. An ordinary failure before entering the
continuation becomes `MIDDLEWARE_ERROR`; a post failure after a downstream result is known is logged
and preserves that result without replay. Synthesized success cannot erase cancellation, unknown
outcomes, or cleanup control. The framework owns result identity, disclosure, and stop facts.

Tool Hooks use the Agent's shared internal dependencies through
`ToolExecutor(..., hook_dispatcher=dispatcher, command_binding=binding)`. Python-only handlers do
not require a command binding; command scripts and stop-receipt draining use the existing binding.
`RuntimeEnvironment` wires both dependencies into its executor. Hooks YAML and public SDK assembly
arguments are not yet exposed; see [iris.hooks](../hooks/README.md) for events and script results.

The order is permission refresh, breaker/cancellation checks, durable effect claim, `tool.before`,
Middleware/body, then eligible `tool.after`. Before rejection or ordinary handler failure produces
`HOOK_REJECTED` or `HOOK_ERROR` and skips Middleware/body/after. These remain error results, but
`ToolErrorPolicy.STOP` does not terminate the Run solely for either code. Other errors and budgets
retain their existing behavior.

After runs only for a body that actually executed and has a known result. It excludes cached
substitutes, preflight failures, cancellation, environment interruption, and unresolved cleanup.
Ordinary body errors remain eligible; `body_status` stays `error` even if Middleware recovers the
result. An ordinary command timeout is drained before after handlers run. An exited body with an
environment stop receipt skips after without changing its existing Run settlement behavior.
The parent subagent path bypasses Hooks; ordinary child tools use their own handlers.

The Dispatcher is the sole source of `hook_feedback`. The final exit overwrites any value supplied
by a tool or Middleware, including when a known result is recovered after cancellation. Real
feedback shares one result block and one artifact projection with the original body. After control
keeps previously collected feedback and commits the known result before settlement. An unknown
after-script action stops remaining handlers and drains command resources: successful draining
logs the supplementary failure without changing the body to unknown; failed draining preserves
cleanup facts. Recovery reuses committed results without replaying handlers.

`CircuitBreaker` tracks consecutive failures by tool name and returns
`CIRCUIT_OPEN` during cooldown.

`DeferredToolIndex` uses a local BM25-like ranker over name, tags, group, and description, with CJK
bigrams, low-weight single characters, query coverage, and stable sorting. Construct `ToolSearchTool`
with a static `ToolRegistryView`, for example `ToolSearchTool(registry.view())`. Its model-facing name
is `tool_search`; `ToolSearchInput(query, include_groups=None, limit=3)` permits limits from 1 to 20.
The lower-level registry search still defaults to 10. Base deny/group/allow and query group filters
apply before ranking and top-k; search results cannot expand the host's group scope.

JSON text and `data["tools"]` contain candidate summaries with `name`, `description` capped at 240
characters, and `group`. No matches return an empty list; full parameter JSON Schema stays out of search text.
A successful system search saves ranked canonical names in the committed tool message's
`metadata.extra.context_revealed_tools`. Executor does not accept this field from other tools as a
disclosure fact. Names come from the successful search body and survive an after-middleware text
replacement; final errors and substitutes for a failed body do not create disclosure.

Search alone does not mutate the registry. With `context_policy.deferred_tools: true`, runtime
registers the tool automatically and selects candidate tool definitions with full parameter JSON Schema
for the next main model request only after the search result commits to that session's history. Newly found tools cannot
be used in the same batch as their discovery call; the model waits for the tool definition in the next
request. See [runtime](../runtime/README.en.md#deferred-tool-definitions) for budgets, forced tools,
and batch recovery.

## Public surface and boundaries

The exact top-level API is `src/iris/tools/__init__.py::__all__`, including
`CallableExecutionMode`, `ToolTimeoutOwner`, `ToolCall`, `ToolNext`, `ExecCommandInput`,
`ExecCommandTool`, and covering models, base/adapters,
registry/view, executor/preflight, permission/artifact/middleware/breaker types, file/human tools,
deferred discovery, schema helpers, and `tool`. Protected `_impl()` hooks and executor private
lifecycle methods are internal.

The package does not run provider loops, persist ordinary session data, or render host UI.
MCP protocol integration lives in `iris.mcp` and uses this package's ordinary tool contracts.

## Maintenance

| Change | Main location | Tests |
| --- | --- | --- |
| Models, callable/schema adaptation, and registration | `base.py`, `schema.py`, `registry.py` | `tests/tools/test_schema.py`, `tests/tools/test_registry.py`, `tests/tools/test_executor.py` |
| Lifecycle and HITL preflight | `executor.py`, `permissions.py` | `tests/tools/test_executor.py`, `tests/tools/test_executor_preflight.py`, `tests/tools/test_human_ask_tool.py` |
| File tools, artifacts, and workspace safety | `builtin/file.py`, `artifacts.py` | `tests/tools/test_file_tools.py` |
| Complete-result storage and current-session reads | `artifacts.py`, `context_access.py`, `../harness/_context_access.py` | `tests/tools/test_middleware_artifact.py`, `tests/harness/test_context_access.py`, `tests/store/test_lifecycle_store_contract.py` |
| Retention declarations and execution-time facts | `base.py`, `executor.py`, `builtin/file.py`, `builtin/web.py` | `tests/tools/test_context_retention.py` |
| Deferred search, static filters, and disclosure facts | `discovery.py`, `registry.py`, `executor.py` | `tests/tools/test_deferred_discovery.py` |
| Circuit breaker | `circuit.py` | `tests/tools/test_circuit_breaker.py` |

```bash
uv run pytest tests/tools
uv run ruff check src/iris/tools tests/tools
```
