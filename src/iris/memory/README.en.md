[中文](README.md)

# `iris.memory`

`iris.memory` is Iris's local long-term-memory SDK. It defines isolation scopes, L1 episodes,
candidates, L2 items, audit events, SQLite persistence, human-readable file projections, explicit
orchestration, and read-only model tools. SQLite is authoritative; Markdown/JSON under
`.iris/memory/` is a projection for humans.

Memory is not currently an `AgentConfig` YAML field and runtime never enables or recalls it by
default. Callers must build a `MemoryService`, inject it into `RuntimeFactory`, and explicitly pass
`RuntimeOptions.memory_query` or `memory_results` for each run.

## Quick start

```python
from pathlib import Path

from iris.memory import (
    MemoryConfig,
    MemoryQuery,
    MemoryScope,
    MemoryWriteInput,
    build_memory_service_from_config,
)

workspace = Path(".").resolve()
service = build_memory_service_from_config(
    MemoryConfig(backend="sqlite"),
    workspace,
)
assert service is not None

scope = MemoryScope(workspace_id=str(workspace), agent_id="notes-agent")
item = service.remember(
    MemoryWriteInput(
        scope=scope,
        text="The user prefers concise answers",
        reason="The user stated this explicitly",
    )
)
results = service.recall(MemoryQuery(scope=scope, text="answer preference"))
bundle = service.build_context(
    MemoryQuery(scope=scope, text="answer preference"),
    max_chars=1000,
)
```

`backend="none"` returns `None` without filesystem effects. Memory root and database paths must
resolve inside the caller-supplied workspace. SQLite FTS5 is used when available; deterministic
text search handles disabled FTS and missing Unicode matches.

## Architecture and lifecycle

```mermaid
flowchart LR
    Input["MemoryObserveInput / MemoryWriteInput"] --> Service["MemoryService"]
    Service --> Store["MemoryStore"]
    Store --> SQLite["SQLiteMemoryStore authoritative data"]
    Service --> Mirror["FileMemoryMirror human projection"]
    Episode["L1 MemoryEpisode"] --> Orchestrator["explicit MemoryOrchestrator"]
    Orchestrator --> Candidate["MemoryCandidate"]
    Candidate --> Item["L2 MemoryItem"]
    Query["MemoryQuery"] --> Service
    Service --> Context["MemoryContextBundle"]
    Context --> Runtime["explicit RuntimeOptions injection"]
```

`MemoryScope` combines workspace, agent, collection, visibility, and optional session. Session
visibility requires a session ID; agent/workspace scopes ignore runtime session IDs so they remain
visible across sessions. `workspace_shared_scope()` uses the stable `__workspace__` / `shared`
convention. Every SQLite operation applies the full scope filter and does not reveal cross-scope
existence.

- `observe()` records an L1 episode and `OBSERVE` event, but no long-term item.
- `remember()` explicitly creates an L2 item and `ADD` event.
- `recall()` returns ranked `MemorySearchResult` objects.
- `forget()` tombstones rather than physically deleting items.
- `MemoryOrchestrator.observe()` uses injected extraction/classification to create candidates.
- `process_candidates()` explicitly accepts, rejects, or promotes candidates; the default no-op
  extractor creates none and no background extraction exists.

Candidate promotion is a single SQLite transaction and is idempotent after success. Item,
candidate, and event IDs are globally unique across scopes.

`MemoryContextBuilder` preserves result order and fits fragments into `max_chars`, truncating only
the first fragment when necessary and counting omissions. Prompt fragments keep semantic metadata
but omit storage source and retrieval score by default.

```python
from iris.runtime import RuntimeFactory
from iris.runtime.models import RuntimeOptions

runtime = RuntimeFactory.from_config_path(
    "agent.yaml",
    memory_service=service,
)
result = await runtime.run_loop(
    "Continue the previous task",
    options=RuntimeOptions(memory_query=MemoryQuery(scope=scope, text="previous task")),
)
```

## Public surface

The large `iris.memory` export surface is grouped as follows:

- models/enums: scope, episode, candidate, item, event, query, search result, and context bundle;
- service/storage: `MemoryService`, `MemoryStore`, and `SQLiteMemoryStore`;
- config: `MemoryConfig` and child models, `build_memory_service_from_config()`, and
  `resolve_memory_path()`;
- orchestration: extractor/classifier protocols, policy, orchestrator, and rule/no-op defaults;
- projection: `FileMemoryMirror` and `MemoryContextBuilder`;
- tools: search/list/get tools, access-policy factories, and `register_memory_tools()`.

The exact set is `src/iris/memory/__init__.py::__all__`. Private SQL helpers, mirror markers, and
tool-payload helpers are not extension contracts.

`register_memory_tools()` exposes only `memory_search`, `memory_list`, and `memory_get`, all with
`READ` capability. There are no model-visible remember/forget tools. Tool input cannot override the
scope; `MemoryAccessPolicy` derives read/write scopes from trusted host context and can include the
workspace-shared scope.

`FileMemoryMirror` creates the fixed Memory/User/Feedback/Reference/Tasks/Sessions projection and
can deterministically rebuild active items plus the most recent 100 events for one scope. It is not
an import source or the audit authority. SQLite uses short-lived connections and wraps storage/JSON
failures as `IrisMemoryError`.

## Current limitations

- no vector database, embeddings, semantic reranker, or remote backend;
- no automatic extraction from session messages or background tasks;
- `MemoryOrchestratorConfig.enabled` is only a config shape and does not construct an orchestrator;
- write-policy config only describes the current `sdk_only`/`tombstone` contract;
- `WorkingMemoryFrame` is reserved and is not consumed by the active runtime;
- collection is a hard SQLite filter, not a managed business object.

## Maintenance

| Change | Main location | Tests |
| --- | --- | --- |
| Models and isolation | `models.py` | `tests/memory/test_models.py` |
| SDK lifecycle and audit | `service.py` | `tests/memory/test_service.py` |
| SQLite, FTS, and transactions | `sqlite.py` | `tests/memory/test_sqlite_store.py` |
| Candidate orchestration | `orchestrator.py` | `tests/memory/test_orchestrator.py` |
| File projection | `mirror.py` | `tests/memory/test_mirror.py` |
| Read-only tools | `tools.py` | `tests/memory/test_tools.py` |
| Runtime injection | `../runtime/memory_context.py` | `tests/runtime/test_memory_context.py` |

```bash
uv run pytest tests/memory tests/runtime/test_memory_context.py
uv run ruff check src/iris/memory tests/memory
```
