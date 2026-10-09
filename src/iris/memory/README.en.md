[中文](README.md)

# `iris.memory`

`iris.memory` is Iris's local long-term-memory SDK. It defines project namespaces, immutable Episodes,
evidence-backed Observations, formal MemoryItems, change events, SQLite persistence, human-readable
file projections, generation stages, and memory tools. SQLite is authoritative; Markdown under
`.iris/memory/namespaces/` is a projection for humans. `MemoryService` is the memory-management SDK
for reads, writes, flush, dreaming, projection, and overview generation.

`memory.enabled` defaults to false. Enabling it automatically adds `memory_search` and
`memory_fetch` and lets the Agent adopt published overviews for a new session or after successful
compaction. The model chooses reads based on its overview and the question; ordinary runs never
search items automatically. Agent construction neither calls the model nor generates an overview.
Enable it with:

```yaml
memory:
  enabled: true
```

When enabled, an injected `memory_service` in `AgentRunner.from_config*()` takes precedence over a
configured SQLite service. When disabled, the injected object is neither mounted, called, nor closed.
CLI and child agents share this assembly path. Each child uses its own switch, read range, and
effective workspace without inheriting the parent's Service. Static `context.yaml` memory slots
remain separate.

## Quick start

```python
from pathlib import Path

from iris.memory import (
    MemoryConfig,
    MemorySearchQuery,
    MemoryWriteInput,
    build_memory_service_from_config,
)

workspace = Path(".").resolve()
service = build_memory_service_from_config(
    MemoryConfig(enabled=True),
    workspace,
)
assert service is not None

item = service.remember(
    MemoryWriteInput(
        text="The user prefers concise answers",
        reason="The user stated this explicitly",
    )
)
response = service.search(MemorySearchQuery(query="concise answers"), ["project"])
for hit in response.items:
    print(hit.snippet, hit.is_complete)
current = service.get_item(item.id, ["project"])
```

`build_memory_service_from_config(config, workspace_root, memory_service=...)` is the sole source
resolver. Disabled memory returns `None` without resolving memory paths or creating files. When
enabled, it returns the injected object unchanged or builds a SQLite service if none was supplied.
An injected service keeps its store, mirror, provider/model, and IO mode. Configured memory root and
database paths must resolve inside the caller-supplied workspace. Default local search uses SQLite FTS5: initialization
and query errors raise `IrisMemoryError`, and no matches return an empty result without LIKE fallback.
New databases use schema version 7; older versions are rejected at initialization without migration,
version overwrite, or deletion. Configured SQLite services use `MemoryIOExecutionMode.THREAD`;
directly constructed services default to `INLINE`, preserving the host's execution choice.
Each THREAD async read or write runs connection setup, SQL, and result construction in one worker job.
Synchronous methods still execute on their caller's thread. Custom stores and synchronous token
estimators should select THREAD only when they support calls from worker threads. INLINE work still
occupies the event loop; maintenance does not override this choice. Independent `MemoryService`
and low-level tool-registration SDK calls do not require the Agent switch.

`MemoryService(..., observability=...)` and the factory's matching parameter accept a host-owned
[observation service](../observability/README.md). Omitting it disables observation without reading
global configuration or creating an exporter. The factory passes raw providers; the constructor
wraps each overview and generation entry point once. Callers should pass unwrapped providers.
An injected complete `memory_service` retains its own observation policy; the runner neither
rewraps nor overrides it. The host closes observation resources after maintenance and other calls finish.
Actual overview, flush, and dream requests carry the `memory_overview`, `memory_flush`, and
`memory_dream` purposes. Within a host Memory maintenance cycle, each actual stage result produces
one `iris.maintenance.result` event. Empty, blocked, and conflict outcomes remain non-errors;
stages without a model request produce no model span. A stage failure marks its maintenance cycle
without changing an already successful model call. Cancellation after commit retains the committed
stage status. Standalone SDK calls and foreground overview generation retain actual model traces
without adding maintenance results to ordinary host spans or creating cycles.
History APIs include `list_episodes`, `list_generation_results`, `get_observation`,
`list_publications`, and `get_publication`, with async counterparts on the service.
Publication records retain actual document bodies after files are overwritten; partial failures and
unconfirmed state commits are distinct from complete publication. Item changes still use MemoryEvent.

`maintain_cycle(namespace, scope=..., cycle_id=...)` returns a `MemoryCycleResult` containing
the actual stage results and remaining work. An empty cycle has no generated results.

## Architecture and lifecycle

```mermaid
flowchart LR
    Input["MemoryObserveInput / MemoryWriteInput"] --> Service["MemoryService"]
    Service --> Store["MemoryStore"]
    Store --> SQLite["SQLiteMemoryStore authoritative data"]
    Service --> Mirror["FileMemoryMirror human projection"]
    Episode["MemoryEpisode source material"] -->|flush| Observation["MemoryObservation with conditions"]
    Observation -->|dreaming| Item["MemoryItem formal knowledge"]
    Query["MemorySearchQuery / item_id"] --> Service
    Service --> Result["Search snippets / Fetch current record"]
    Service --> Overview["explicit refresh_overview"]
    Overview --> Window["adopted system overview window"]
    Result --> History["ordinary tool-result history"]
```

Each workspace has its own database/service. Items use an opaque `namespace` string, defaulting to
`project`; agents reading that namespace share project memory. Different workspaces use different databases.

`service.search(MemorySearchQuery(query="..."), ["project", "notes"])` searches allowed namespaces in one
query and applies the limit after global ranking. `get_item(item_id, namespaces)` and
`list_items(namespaces)` also accept a combined read range. Writes and generation operations use one
namespace. An empty read range returns no items.

- `observe()` records an immutable `MemoryEpisode` and `OBSERVE` event, but no long-term item.
- `await flush()` extracts evidence-backed `MemoryObservation` records and advances source cursors.
- `await dream()` reconciles observations and explicit changes into formal Items through additions,
  updates, merges, retirement, or supporting evidence.
- `remember()` explicitly creates a formal Item and `ADD` event without requiring extraction first.
- `update(item_id, namespace, patch, reason=...)` updates an item without changing its ID.
- `search()` returns `MemorySearchResponse(items, has_more)`, with identity fields and a raw snippet per hit.
- `forget()` tombstones rather than physically deleting items.

Episodes contain source `records` with stable IDs. Their content and Observations stay immutable;
the store separately maintains flush cursors and observation processing states. Observation
applicability guides dreaming, while formal Item text includes the conditions needed to use its
knowledge. Search, Fetch, and overviews read only active Items, not Episodes or Observations.

Generation requests use logical `response_format="json_object"`; the selected provider adapter
handles protocol encoding.

Flush selects information useful for future tasks: preferences, corrections, project conventions,
reusable experience, and important pending work. Chitchat, routine activity, and temporary requests
with no future value may produce no observations. Dreaming checks drafts against original evidence
with roles, timestamps, and metadata, then combines related points into concise Items. Text may omit
secondary details while retaining the subjects and conditions needed to avoid misuse. Dates, versions,
attribution history, and untried alternatives are included when useful; evidence links retain the history.
Compression must not invent facts or strengthen conclusions: one experience does not establish a
general rule, explicit default preferences retain their scope, and unverified does not mean ineffective.
Details the user explicitly asks to remember are retained. The `reason` briefly states the purpose of
recording or consolidating. These are model instructions: JSON schema validation checks structure and
field constraints, not whether the text's conclusions follow from its evidence.

Default Flush and Dream strategies ship in [`memory_flush.j2`](../prompts/memory_flush.j2) and
[`memory_dream.j2`](../prompts/memory_dream.j2). Runnable Agent initialization adds missing seeds to
the project's `prompts.root` (default `.iris/prompts`) while preserving existing text. Generation
reads only the explicitly bound `MemoryService.prompt_source`. Edit strategies in that project
directory; templates preserve quotes and `<>&` as plain text. Domain code appends fixed business
instructions and the response model's JSON Schema to every request. Editable text cannot remove
the actual output contract; responses still pass through the existing single parsing boundary.
Source loading and template rendering failures become `IrisMemoryError`.

An automatic cycle takes one in-memory source snapshot for Flush, Dream, and Overview; edits during
the cycle take effect next cycle. Each standalone `flush/dream/refresh_overview` call takes a fresh
snapshot, including template dependencies. Snapshots are not archived. Configured services receive
the root project's source; injected services keep their host binding and runners do not rewrite it.
Ordinary reads, search, and remember/update/forget require no source. Standalone generation requires
an explicitly initialized `PromptSource`, failing at the domain boundary when absent instead of
inferring a working directory.

Flush/dream requests use `temperature=0` and `response_format={"type": "json_object"}`; the generation
provider must support both parameters. JSON mode constrains response format; the existing parsing
boundary still checks fields and evidence references. Invalid responses are not committed, and saved
inputs remain available for retry. Low randomness and comparison with original evidence do not
guarantee semantic correctness.

`MemoryEvidenceRef` points to a record span within an Episode or a real explicit-write event.
An Item's `evidence` supports its current text; observation resolutions and MemoryEvents retain the
historical explanation. A semantic text update replaces current support with this write event and
evidence explicitly supplied by this call. Classification or metadata-only updates keep existing
support. Write tools encode `[lifecycle_source_id, run_id, call_id]` as compact JSON in the existing
`source_id`, distinguishing repeated provider call IDs across runs and lifecycle sources without
claiming user confirmation.

Omitted patch fields remain unchanged. Use `[]` and `{}` to clear artifacts and metadata; all patch
fields reject explicit `null`. Updates and soft deletes acquire a `BEGIN IMMEDIATE` write lock before
reading current data. Text, current evidence, events, pending changes, and revision commit together.
Changes to different fields from separate connections merge sequentially.

Flush atomically commits observations and source progress, including progress for an empty extraction.
Dreaming reads fixed inputs, related items, and corrections in one snapshot. Model calls run outside
the transaction; the commit compares revisions and applies the entire plan and input resolutions.
The Flush request sends source information and the known run outcome once per Episode, linking excerpts
with short labels. Durable Episode/Record IDs, message boundaries, and block ordinals stay in
the program; tool status and explicit memory targets remain available to the model.
The model receives a compact projection of observations, related items, changed event fields, short
evidence refs, and original excerpts. The program retains durable Episode/Record/Event locators;
request-local record labels and character spans let the model recognize overlapping evidence.
Conflicts leave inputs unconsumed. Capacity-blocked inputs remain available for retry.
`generation_state()` exposes backlog, blocked inputs, and stage results. Configure
`generation_provider`, `generation_model`, and `generation_config` on the service. Standalone SDK
callers can explicitly run `await service.flush("project")` and `await service.dream("project")`.

### Automatic generation and background lifecycle

Reading alone does not enable generation costs. Opt in explicitly:

```yaml
memory:
  enabled: true
  write_namespace: project
  generation:
    enabled: true
maintenance:
  idle_seconds: 300
  min_pending_runs: 10
```

`generation.enabled` defaults to false. Configured services reuse the Agent's resolved provider and
model. Injected services retain their own generation dependencies and budgets. Automatic operation
requires generation provider/model, overview provider/model, and a mirror; constructing the runner
without these dependencies raises `IrisConfigError`.

The host owns one shared `MaintenanceCoordinator` and explicitly binds each root runner's lifecycle
reader and `MemoryMaintenanceBinding`; factories do not create private maintenance schedulers.
`maintenance.idle_seconds` defaults to 300 and accepts zero; generation budgets and policies belong
to `memory.generation`. See [harness](../harness/README.en.md) for host wiring.
An automatic-maintenance runner without a binding fails at its first preparation/run boundary.

A new automatic Episode batch requires both the quiet interval and `maintenance.min_pending_runs`
eligible new Runs, defaulting to 10. Each database/namespace counts independently; multiple Episodes
from a Run count once, and consumed or entirely filtered input does not count. Admission is durable:
budget-limited remainders continue in later cycles or after restart, while new Runs form the next batch.
Time alone never waives the count; use a manual cycle or min_pending_runs=1 for earlier processing.

Capture promptly records committed source text without waiting for the learning count. Automatic learning selects only terminal,
fully captured runs. A WAITING session excludes its own earlier material, while other sessions can
continue. Tool changes resolve complete call identities through captured records; incomplete or unmatched changes
remain pending. Explicit SDK content without a Run keeps its existing semantics.
`list_pending_sources()` exposes source references. The host supplies all three required
`MemoryMaintenanceScope` fields: `allowed_sources`, `check`, and `episode_sources`. Only Episode flush
and its remaining count use episode_sources; observations, explicit changes, and retry_blocked keep
allowed_sources. Explicit observe input without a Run is not excluded by the Episode source set.
The store filters before limits, including reselection and counts; generation checks actual batch
sources before model calls and input commits. Published knowledge remains available for comparisons,
projections, and overview repair.

The store's `read_learning_readiness(namespace)` returns `MemoryLearningReadiness` using only source
identities, completion, admitted, and remaining-content columns, without reading Episode, Observation,
or Event payloads. Multiple Episodes from one Run form one candidate; has_unsourced identifies explicit
input without a Run. The snapshot also includes item_revision, projection_revision, and a
has_pending_derived hint for pending observations or changes; the hint does not replace source checks.
The service exposes `await aread_learning_readiness(namespace)`.
`has_retryable_derived(namespace, *, budget, allowed_sources=None)` also finds blocked downstream
inputs whose budget has changed. Without a source scope it reads short state only; with one it
applies the existing source qualification query. The service's
`await ahas_retryable_derived(namespace, *, allowed_sources=None)` uses the current dream budget.
Both are read-only; `retry_blocked` inside the maintenance cycle still reopens the inputs.
`admit_learning_sources(namespace, allowed_sources=..., threshold=...)` rereads candidates in a write
transaction. When at least threshold eligible, fully captured new Runs still have evidence text, it
persistently admits all current candidates. The async service entry point is `aadmit_learning_sources`.
Capture and consumption preserve admission across reopening. Remaining content follows the character
cursor. Empty or evidence-disabled prefixes advance without model budget estimation or model calls,
while retaining source checks and CAS. The host selects the admission threshold; these read APIs do
not start maintenance.

Existing observations, explicit changes, projection/overview repair, and explicit observe input without
a Run do not require ten new Runs. Empty sources can settle without admission or model calls. Manual
`request_memory_cycle()` bypasses the automatic time and count gates while retaining foreground,
eligibility, and locking rules. Standalone SDK flush/dream/refresh_overview remain explicit operations.
When only material quantity blocks progress, the resource reports waiting_for_materials. Snapshot
pending_new_runs/min_pending_runs come from the last async check, without synchronous database reads;
next_eligible_at remains unknown.

After the quiet interval, the coordinator acquires the database/namespace OS lock and calls
`maintain_cycle(namespace, scope=..., cycle_id=...)`: dream existing observations or changes first; otherwise flush,
then dream, then repair projections and the overview. Each cycle is bounded and returns whether
eligible work remains. New foreground input cancels uncommitted generation without waiting for the
model; the task slot and lock remain held until actual synchronous work finishes. Closing one runner
detaches only its binding and captures remaining text. The host drains the coordinator before closing
shared resources; close never starts learning.
Automatic maintenance owns a dedicated single-thread worker for THREAD services. Only the maintenance
task sends synchronous work there: background IO, prompt rendering, flush selection, dream/overview request construction,
token estimation, and response parsing. Foreground reads/writes, source registration, and Capture
retain their existing execution path, so they do not queue behind background computation. Standalone
SDK calls do not implicitly create a maintenance worker. `provider.complete()` remains async on the
caller's event loop. Selection checks cancellation between records; no new cycle starts until old
synchronous jobs have actually exited. Late results do not start another model request or commit.
Shutdown waits for real IO and computation, then releases the worker.

The execution and cancellation primitive lives in `iris.utils.generation_worker`. Memory and
project evolution own separate worker instances and resource locks. Evolution neither reads private
Memory storage nor waits for a particular Memory model call to provide its input.
The shared `BackgroundIO` implementation tracks THREAD jobs, collects short-commit receipts after
cancellation, and drains pending work. Each service owns a separate instance; Memory still owns the
INLINE/THREAD choice.

`generation_prompt_descriptions()` and `overview_prompt_description()` expose fixed contracts shared
with actual requests, plus representative template variables. Assembly can bind these to finite prompt
revision without exporting private response models or duplicating schemas. Real outputs still pass
through Memory's parsing and application boundaries.
Capture reads at most 128 messages per page and yields between pages, sealing only at the full run
cutoff. SQLite releases its read transaction and lifecycle lock after fetching raw rows, before decoding
messages. Registration and Capture before a run returns still await durable storage. A dedicated worker
isolates the background queue, but does not eliminate database locks or CPU contention, nor guarantee
zero foreground overhead.

SQLite lifecycle sources have a persistent UUID and bounded run-message ranges, so restart can capture
missing suffixes without learning fork history twice. Without the original lifecycle reader after an
InMemory restart, captured material remains pending; automatic continuation requires SQLite lifecycle.
Child agents retain reading and explicit writes, but their
internal traces are not automatically collected. BCI, reasoning, and memory readback text are excluded
from new evidence; observations reference stable records and half-open character ranges.
An automatically captured Episode keeps the run ID in top-level `source_id` and the lifecycle source
ID in metadata. Pending-Episode reads use those fields to attach the source's final outcome.

Flush and dream each default to 32,000 input tokens and 4,000 output tokens. Configure these through
`flush_input_budget_tokens`, `flush_output_budget_tokens`, `dream_input_budget_tokens`, and
`dream_output_budget_tokens` under `generation`. Flush splits long records into fixed spans. Dream
budgets complete comparison packages, preserving oversized inputs as blocked while processing
unrelated work. Dependency changes, a rebuilt Agent with a changed budget, or an explicit
`await service.dream(namespace, retry_blocked=True)` make blocked inputs eligible again.

Model failures wait for new activity or restart instead of retrying in a tight loop. Projection failures
do not re-extract consumed material: later maintenance repairs category files by projection revision
and overviews by overview revision. Maintenance usage is stored in `GenerationResult`, separate from
foreground run usage and model steps. `generation_state(namespace)` / `await ageneration_state(namespace)`
reports pending/blocked counts, item/projection/overview revisions, and the latest result per stage.
Flush results expose exact committed spans in `consumed_ranges`.
Dream reports operation counts, `processed_observations`, `processed_changes`, and `unchanged`.
The last count refers to inputs whose target was not added, rewritten, merged, or deleted; supporting
evidence without a text change is included.

The independent SDK can run each stage explicitly without enabling Agent automation or requiring a
mirror:

```python
from pathlib import Path
from iris.memory import MemoryObserveInput, MemoryService, SQLiteMemoryStore
from iris.prompts import PromptSource

workspace = Path.cwd()
service = MemoryService(
    SQLiteMemoryStore(Path("memory.db")),
    prompt_source=PromptSource.initialize(workspace),
    generation_provider=provider,  # The application's configured CompletionProvider.
    generation_model="your-model",
)
service.observe(MemoryObserveInput(text="This project now uses uv for dependencies."))
flushed = await service.flush("project")
dreamed = await service.dream("project")
state = service.generation_state("project")
```

Each call processes one batch; `has_more` reports pending inputs for that stage. Dream does not
implicitly flush. Explicit `refresh_overview()` still requires its own provider/model and mirror.
Background publication does not replace an adopted session window; adoption remains limited to a new
session or successful compaction.

### System overview window

On the first session input and successful compaction, runtime loads published overviews and chooses
full core facts plus knowledge scope, or the complete knowledge scope alone. The selected window is
committed atomically with the session transition. Tool loops, new runs, HITL, and recovery retain the
adopted window; failed compaction does not replace it. The overview belongs in system context,
while Search/Fetch results enter ordinary tool history.

All namespaces, warnings, tool instructions, and wrapping share a budget of
`floor(compaction.input_budget_tokens * memory.overview.system_budget_ratio)`, with a default ratio
of `0.02`. If full content does not fit, runtime tries the complete knowledge scope. If that also
exceeds its budget, it reports a capacity error instead of cutting topics. Selection estimates actual
request tokens and retains the complete system character limit.

Model instructions treat topics absent from the adopted overview as unavailable and do not search
for them. Without an overview, normal chat continues without long-term-memory reads for this window.
An overview covering new topics must be published, then adopted in a new session or
after successful compaction. Already covered topics may still query current database records. This
is a model instruction, not a database topic filter. Search does not require a subsequent Fetch.
Updates and forgetting never rewrite previously saved conversation history.

The switch is fixed at Agent construction; same-session hot switching is not supported. Rebuild the
Agent and start a new session after changing configuration. Disabled requests omit the memory
addendum even if a saved window contains one, without deleting that window, the database, overview
files, or historical tool results. Static memory, ordinary history, and summaries remain intact.
Normal successful compaction may still commit an empty overview window. Re-enabling and reusing an
old session does not force a new overview; start a new session to adopt the current publication.

## Explicit overview generation

The host can call `await service.refresh_overview(namespace)` to summarize all active formal Items in
that namespace, grouped by category/kind, into core facts and knowledge scope. The complete result
is published as `Memory.md` in the canonical namespace directory. Configure `overview_provider`,
`overview_model`, and `overview_config` on the service; the provider follows
`iris.providers.CompletionProvider`. Agent configuration binds the resolved main provider, while
an explicitly injected service keeps its host-supplied configuration. Construction and reads do not
call the model; optional runner maintenance owns automatic generation.

The default Overview strategy ships in [`memory_overview.j2`](../prompts/memory_overview.j2);
generation reads the project's source snapshot. Request assembly appends the two-field schema from
`MemoryOverviewContent` and its fixed business instructions. Editing the project strategy does not
require changing input preparation, budgets, or response parsing.

```python
# The service already has its overview provider/model configured.
result = await service.refresh_overview("project")
documents = await service.aload_overviews(["project"])
```

`MemoryOverviewConfig` defaults to a 96,000-token generation input budget and 4,096 output tokens.
An oversized complete snapshot fails without truncation or batching. One normally completed JSON
response must contain `core_facts` and `knowledge_scope`; facts may be empty, scope must not be.
Invalid or incomplete responses and publication errors preserve the previous complete file, with
known model usage attached to error context. An empty active Item snapshot publishes “当前无记忆”
without a model call.
Successful, failed, and cancelled overview attempts persist a separate `GenerationResult` with
their known usage and publication outcome, available through `generation_state()`.

Generation runs outside the publication lock. A short locked source-version comparison prevents an
older generation from replacing a newer overview. A candidate may still publish behind current
items and receive a stale warning. `load_overviews/aload_overviews` accept only the complete new
format; old formats require explicit refresh. A missing file returns a fixed missing-overview
message without scanning items or generating a directory. `MemoryOverviewDocument.navigation`
contains knowledge scope. A service without a mirror returns no documents and rejects refresh as
missing generation dependencies.

Generation budgets are independent of main-request window budgets; `system_budget_ratio=0.02`
controls the system window selection described above.

## Public surface

- Inputs and results: [MemorySearchQuery, MemorySearchHit, and MemorySearchResponse](models.py),
  plus Episode, Record, Observation, Item, EvidenceRef, Event, and write/update models.
- Service and storage: [MemoryService](service.py), [MemoryStore](store.py), and
  [SQLiteMemoryStore](sqlite.py). Synchronous `search/get_item/list_items` remain available for SDK
  reads. `asearch/aget_item/alist_items` adapt each complete operation. Writes, event reads, and the
  extraction SDK remain available.
- Overview: `MemoryOverviewConfig/Content/Document/GenerationResult`, `refresh_overview()`,
  `load_overviews()`, and `aload_overviews()`.
- Configuration: [MemoryConfig](config.py), `build_memory_service_from_config()`, and
  `resolve_memory_path()`.
- Generation: [flush / dream](generation.py), [stage models and configuration](generation_models.py),
  and `generation_state()`.
- File projections: [FileMemoryMirror](mirror.py) and [MemoryFileAccess](files.py).
- Tools: [Search/Fetch and Remember/Update/Forget](tools.py), policy factories, and explicit registration.

The complete export set is [__all__](__init__.py). Private SQL, lexer, and payload helpers are not
SDK extension protocols.

See [memory organization and adoption (Chinese)](../../../docs/design/memory.md) for Search/Fetch,
overview windows, and generation responsibilities. Ordinary retrieval uses SQLite FTS with explicit
required-phrase filtering; optional Decision recall is described below. No vector database is required.

### Plain-text search

```python
query = MemorySearchQuery(
    query="answer preferences",
    required_terms=["concise"],
    categories=["user", "feedback"],
    kinds=["preference", "correction"],
    limit=8,
)
response = await service.asearch(query, ["project"])
```

`query` is required and must remain nonempty after trimming surrounding whitespace.
`required_terms` defaults to empty, adding no required body phrases.
Categories and kinds default to empty, meaning no filter for that dimension.
The result limit defaults to 8 and accepts `1..100`. Unknown fields are rejected, and namespace is
not part of model input. Storage filters allowed namespaces, categories/kinds, and active status
before ranking by BM25 ascending, then updated_at/id descending. It reads `limit + 1`, returns only
limit hits, and computes `has_more`. Values within one filter dimension are OR; dimensions are AND.
Only `MemoryItem.text` is indexed. Active Items are searchable; Episodes, Observations,
and deleted/superseded items are excluded from results.

Indexing, queries, and raw-text positions use the same lexer: lowercase ASCII letter/digit runs,
adjacent bigrams for Chinese runs, and a single character only for an isolated Chinese character.
Punctuation and underscores separate tokens. Query terms are deduplicated in first-occurrence order
and quoted as literal OR terms. Neither input text nor query terms are truncated; indexing retains
all terms and frequencies. A nonempty query with zero terms, no matches, or an empty read range returns
`MemorySearchResponse((), False)`, never recent items.

The model or SDK can supply `required_terms` explicitly. The same item's body must match the ordinary
query's OR group and every required phrase. Each phrase uses the same lexer and requires its tokens
to be adjacent and ordered, without deduplicating repetitions. For example:

```text
query="rollback threshold", required_terms=["Clearport", "billing export"]
→ ("rollback" OR "threshold") AND "clearport" AND "billing export"
```

Separate phrases have no relative order or distance requirement; tokens within a phrase do, so
`go go` requires two consecutive `go` tokens. This is token matching, not a literal substring search.
English case is ignored, but Chinese punctuation can change the bigram sequence: “账单-导出” does
not match the phrase “账单导出”. Required phrases have no 64/128-term cap. A phrase with no indexable
tokens is rejected by `MemorySearchQuery`. An ordinary query with no tokens still returns nothing;
required phrases alone are not a browsing API. No conditions are silently removed or relaxed when
there are no matches. This reuses the existing FTS5 index, without a schema or index migration.
Use category/kind filters only when the stored labels are known. A name appearing in a body does not
establish that its facts apply to that entity; callers must check applicability.

Each hit contains exactly `item_id/namespace/category/kind/snippet/is_complete`. Bodies of at most
300 Python Unicode characters are returned in full. For longer bodies, the first matching token's
start h gives `start=max(0,min(h-150,len(text)-300))`; the snippet is the exact 300-character slice,
without ellipses or highlighting. `is_complete` describes body completeness, not verified relevance.
Identical text under different IDs remains separate.
Snippet placement still follows the first ordinary-query hit. A required phrase may lie outside the
snippet; Fetch can supply the remaining body when needed.

### Optional direct semantic recall

Keep the Agent's `memory.enabled: true` and enable `memory.recall: true` in its separate Decision
file. This feature does not require tool discovery. See [Decision SDK and configuration](../decision/README.md)
for the file reference and credentials. The switch changes only the `memory_search` retrieval backend:
both modes expose the same five input fields, description, and `items/has_more/hint` output.
Ordinary SDK `service.search/asearch` calls continue to use local lexical search.

At each call, enhanced Search obtains the allowed namespaces and performs one
`alist_items(namespaces, limit=None, categories=..., kinds=...)` read of all matching ACTIVE formal
memories. It then applies `required_terms` with the same ordered, adjacent token-phrase semantics
above. It never infers hard conditions from query or narrows candidates by query text, BM25, or output
limit. [_query.py](_query.py) owns phrase matching; [recall.py](recall.py) owns scoring and selection.
An empty filtered set skips the evaluation; a single candidate still needs a score.

One Decision request sends a state containing only `query` and an ordered `memories` array of body
strings. Each body receives one four-level Score: 0 means irrelevant or explicitly inapplicable;
1 means background only; 2 means partial usable evidence; 3 means direct answering evidence with
visible applicability conditions satisfied. Only the service's `score >= 2.0` qualifies. Results sort
stably by descending score, preserving database updated_at/id descending order for ties, then take
limit hits. `has_more` means the qualifying count exceeds limit; repeating a request does not advance
a page. The threshold is a current business rule, not a confidence threshold or a verified quality guarantee.

The request excludes local IDs, namespaces, classifications, timestamps, metadata, required_terms,
and limit. Bodies are not repeated in each question. Selected hits return the full original body in
`snippet` with `is_complete=true`, even without lexical overlap with query. Ordinary result character
budgets and artifacts still apply. Scores and probabilities stay out of the result body. A successful
actual evaluation adds `metadata["decision"]` with `feature="memory.recall"`, provider, actual model,
question count, and token usage. Capacity and service failures become `IrisMemoryError`; there is no
silent truncation, batching, lexical fallback, or empty-success replacement. Outer cancellation propagates.

SDK callers may borrow an evaluate-only object through `MemorySearchTool(..., decision_client=evaluator)`
or `register_memory_tools(..., memory_decision_client=evaluator)`. Both also take
`prompt_snapshot=prompt_source.snapshot()` to freeze `memory_recall_instruction` when constructing
the tool. Calls still use current candidate indices and bodies; new tools adopt strategy edits.
Code owns score levels, threshold, question IDs, and state keys. Local Search needs no prompt source.
Search neither creates nor closes
the client and never stores its mode on the shared `MemoryService`. Fetch and write tools do not
receive it, so two Agents may share a service while selecting different search modes. The environment
closes Agent-owned connections; the host owns injected clients. Only enhanced Search declares
`READ+NETWORK`, following the built-in default allowance and ordinary custom permission decisions.
## Memory tools

Enabled Agents automatically register tools in Search, Fetch order. Both default to `READ`; a Search
using Decision recall is `READ+NETWORK`, while Fetch stays `READ`. Do not declare
`memory.search` or `memory.fetch` in `tools.builtin`: registry assembly rejects those names with
`IrisConfigError` and points to `memory.enabled`. The former `memory.backend` field is rejected at
configuration parsing. `include_tools=False` still omits schemas from that request, with overview
instructions based on the tools actually available.

Low-level `register_memory_tools()` defaults to no tools and accepts explicit
`memory.search/fetch` selection in SDK code. Agent YAML uses `memory.enabled` to register read tools.
Search directly uses `MemorySearchQuery` as its input and returns `items` and `has_more`. Only when
more candidates exist does it add the hint “还有候选；这不要求继续查询。” Stop when snippets provide
enough evidence; search again only for missing necessary information, not merely to rephrase a
query without a new lead.

Fetch takes one nonblank `item_id`; a known ID can be fetched without a preceding Search. It calls
current `aget_item()` and returns `{"item": item.model_dump(mode="json")}` with every stored field,
including complete text, source, current evidence, metadata, artifact references, status, and timestamps.
Raw evidence and attachments are not opened. Missing, inactive, and out-of-range items report “允许读取范围内未找到有效记忆”. Repeated
Fetch calls are not suppressed. Updating an item after Search means a later Fetch returns its new
value. Ordinary `max_result_chars=50000` and ToolExecutor artifact handling still apply.

Agents that write memory explicitly declare `memory.remember/memory.update/memory.forget`, exposing
`memory_remember/memory_update/memory_forget`. These retain the normal `WRITE` permission, claim,
and result-commit flow and share the SDK service. Agent write declarations require enabled memory;
enabling memory does not register writes automatically. Forget reports whether a soft delete actually
occurred. A stale projection adds a warning while retaining the committed database success. For example:

```yaml
memory:
  enabled: true
  read_namespaces: [project, notes]
  write_namespace: project
tools:
  builtin: [memory.remember]
```

`MemoryAccessPolicy(read_namespaces=[...], write_namespace=...)` binds host-owned read/write ranges,
defaulting to `project`. Its factory runs for every tool call; tool input cannot override namespace.
Policy evaluation stays on the event loop; in THREAD mode, the database operation uses one complete
service worker job. SDK registration selects builtin names with
`register_memory_tools(..., tool_names=("memory.search", "memory.fetch"))`.

## File projections and persistence

`FileMemoryMirror.initialize_layout()` only creates directories. Canonical category documents live
under `namespaces/ns_<base64url of namespace UTF-8>/`, covering User, Feedback, Reference, Tasks,
and Sessions. It does not create `Memory.md` or rewrite legacy root files. Each entry keeps its raw
Markdown first, followed by complete metadata inside `<details>`. These are human-readable views,
not a record parsing or reverse-import protocol. Episodes, Observations, and Events remain in SQLite.

After a committed write, `rebuild_from_store()` uses `publish_projection()` to read a complete active
Item snapshot within a short write transaction and atomically replace each category document. Only
successful publication of every document advances `projection_revision`. `item_revision` advances
with effective formal-knowledge changes in the same transaction as Items, FTS, current evidence,
events, and input processing state;
an unchanged patch does not rewrite the item, event, or revision. A partially failed publication
keeps the database result successful and leaves the projection version behind. Write-tool results
include a warning; ordinary file tools use `MemoryFileAccess` to report freshness from the actual
file source revision. Database queries remain available when publication fails. Explicit rebuild
errors still propagate. Configured SQLite services always maintain these
category projections; a directly constructed SDK service may omit the mirror.
SQLite uses short-lived connections and wraps storage/JSON failures as `IrisMemoryError`.
FTS contains all item states, while Search and Fetch only return active records. Management SDK
`store.list_items(..., include_deleted=True)` can inspect soft-deleted records.
Item/index writes are transactional; `rebuild_index()` can rebuild from the authoritative table.
Public store `list_items()`, `list_events()`, and `list_observations()` calls reject limits outside
`1..100` with `IrisMemoryError` instead of silently clamping them. `list_items(limit=None)` requests
a complete active read for mirror projections and direct semantic recall.

## Current limitations

- no vector database, embeddings, semantic reranker, or remote backend;
- InMemory lifecycle cannot recover source eligibility after exit; captured material remains pending.
  Automatic continuation across restarts requires SQLite lifecycle;
- namespaces are database-local grouping strings, not a separate management service.

## Maintenance

| Change | Main location | Tests |
| --- | --- | --- |
| SDK lifecycle, namespace reads, and search results | `models.py`, `service.py`, `sqlite.py` | `tests/memory/test_service.py` |
| Concurrent commits, field updates, FTS completeness, and search filters | `sqlite.py`, `_query.py` | `tests/memory/test_sqlite_consistency.py` |
| Async IO, combined tool reads, query terms, and query plans | `service.py`, `tools.py`, `sqlite.py`, `_query.py` | `tests/memory/test_async_io.py`, `tests/memory/test_tools.py`, `tests/memory/test_query.py`, `tests/memory/test_search.py`, `tests/memory/test_sqlite_query_plan.py` |
| Namespace snapshots, projection revisions, and atomic replacement | `mirror.py`, `files.py`, `sqlite.py` | `tests/memory/test_mirror.py`, `tests/memory/test_revisions.py` |
| Explicit overview generation, loading, and versioned publication | `overview.py`, `service.py`, `mirror.py` | `tests/memory/test_overview.py`, `tests/memory/test_async_io.py` |
| Flush / dreaming and atomic input consumption | `generation.py`, `generation_models.py`, `sqlite.py` | `tests/memory/test_generation.py`, `tests/memory/test_generation_store.py` |
| Overview-window adoption, compaction, and recovery | `../runtime/runtime.py`, `../runtime/memory_context.py` | `tests/harness/test_auto_memory.py`, `tests/harness/test_runner_memory.py`, `tests/runtime/test_memory_context.py` |

Run targeted tests from the repository root:

```powershell
uv run pytest tests/memory
uv run ruff check src/iris/memory tests/memory
```

Guides and reference (Chinese): [Memory usage](../../../docs/cookbook/memory.md) · [Memory organization and adoption](../../../docs/design/memory.md) · [Memory reference](../../../docs/reference/memory-goals.md).
