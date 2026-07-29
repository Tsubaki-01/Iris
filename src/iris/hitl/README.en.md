[中文](README.md)

# `iris.hitl`

`iris.hitl` defines JSON-safe domain models and a stateless projection service for human-in-the-loop
gates. Permission confirmations and human questions share an exact tool-call identity while keeping
their distinct prompt and response semantics.

The package does not implement UI, provider calls, tool execution, or persistence transactions. On
the new lifecycle path, `iris.lifecycle.LifecycleStore` is the only authority for run, checkpoint,
tool-call, and interaction writes. `HumanInteractionService` only constructs, validates, and
projects values.

## Core models

- `ToolCallSnapshot` contains the tool ID, name, arguments, workspace, and stable SHA-256
  `fingerprint`.
- `PermissionPrompt` and `PermissionInteractionResponse` model approve/reject for one exact call.
- `QuestionPrompt` and `QuestionInteractionResponse` model a question, optional choices, and a
  non-empty answer.
- `HumanInteractionRequest` is the only `tool_call + typed prompt` envelope.
- `HumanInteraction` is a durable fact bound to a run, session, step, and tool call. Its new
  lifecycle is `pending -> resolved -> closed`, with an optional aware `expires_at`.
- `ApprovedToolCall` is a frozen projection DTO for approval; it does not authorize a side effect
  by itself.

The `consumed` status and `InteractionResumePhase` remain temporarily for old-runtime
characterization before Phase 5 removes them. They are not part of the new `AgentRunner`
interaction flow.

## Stateless service

```python
from iris.hitl import HumanInteractionService

service = HumanInteractionService()
```

`HumanInteractionService` has a zero-argument constructor and owns neither a store nor a clock. It
provides:

- `create_pending(request, *, run, step_index, expires_at)` to construct an unsaved interaction
  from an active run snapshot;
- `validate_response(interaction, *, run, response, now, environment_fingerprint)` to validate the
  waiting identity, response kind, expiry, and environment fingerprint;
- `project_response(interaction, response)` to project a resolved question to an answer
  `ToolResult`, a rejection to `USER_REJECTED`, or an approval to `ApprovedToolCall`.

Every method is persistence-free. After approval, the runtime still rechecks current policy and
must claim the exact prepared call through the lifecycle store before the tool effect.

## Resume flow

A host renders `RunResult.pending_interaction`, collects a typed response, and calls:

```python
result = await runner.resume(
    run_id,
    interaction_id=interaction.interaction_id,
    response=response,
)
```

`AgentRunner.resume()` loads the waiting run and exact interaction, lazily settles an elapsed run
deadline or interaction expiry, then uses the stateless service to validate the response. The
lifecycle store persists the resolved response with CAS, closes the old interaction, and creates a
new activation for the same `run_id`. Execution resumes through the same engine from checkpoint v1.
Pure reads never settle expiry implicitly.

Retrying the same durable response is idempotent. A different response, wrong run/interaction/kind/
version/fingerprint, or environment drift fails closed. Follow-up gates in one batch are exposed in
their original order, with at most one open interaction per run.

## Public API and maintenance

The package exports typed prompts/responses, `HumanInteraction`, `ApprovedToolCall`,
`HumanInteractionService`, and `make_call_fingerprint()`. `InteractionStore` and
`InMemoryInteractionStore` are still exported temporarily for old-runtime compatibility. New code
must use `iris.lifecycle.LifecycleStore` and must not combine or dual-write the two storage paths.

| Change | Main location | Tests |
| --- | --- | --- |
| JSON-safe models and state constraints | `models.py` | `tests/hitl/test_models.py` |
| Create/validate/project semantics | `service.py` | `tests/hitl/test_service.py` |
| Aggregate interaction transactions | `../store/in_memory.py`, `../store/sqlite.py` | lifecycle store contract |
| Durable resume and expiry | `../harness/runner.py` | runner resume/expiry tests |

```bash
uv run pytest tests/hitl/test_service.py tests/harness/test_runner_resume.py tests/harness/test_runner_interaction_expiry.py
uv run ruff check src/iris/hitl tests/hitl
uv run mypy src/iris/hitl
```
