[中文](README.md)

# `iris.hitl`

`iris.hitl` contains only typed human-in-the-loop domain models and the stateless
`HumanInteractionService`. It stores no interaction, owns no clock, and executes no tool. The same
`LifecycleStore` commits all durable interaction facts together with the run aggregate.

## Domain models

- `ToolCallSnapshot` captures exact call identity, arguments, workspace, and SHA-256 fingerprint.
- `PermissionPrompt` and `QuestionPrompt` represent the two human request kinds.
- Responses are typed by request kind.
- `HumanInteractionRequest` binds a tool subject to a prompt.
- `HumanInteraction` has `pending | resolved | closed` state, version, and timestamps.
- `ApprovedToolCall` is the exact approval projection passed to the engine.

The standalone interaction store, consumed/resume phases, checkpoint payload, and stateful service
have been removed.

## Stateless service

`HumanInteractionService` only constructs a pending value, validates a response against exact
run/interaction/environment facts, and projects the response to either `ToolResult` or
`ApprovedToolCall`. It performs no persistence. Harness uses lifecycle commands for atomic state
transitions.

## Fingerprint

`make_call_fingerprint()` hashes canonical JSON for session/run/call/tool/arguments/workspace.
Approval applies to that exact subject only; identity or environment drift fails closed.

## Public API

`iris.hitl` exports the typed models, enums, fingerprint helper, and stateless service. It exports
no interaction store or compatibility adapter.

## Verification

```bash
uv run pytest tests/hitl tests/harness/test_runner_resume.py
uv run ruff check src/iris/hitl tests/hitl
uv run mypy src/iris/hitl
```
