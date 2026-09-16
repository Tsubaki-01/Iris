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

Public `SubagentExpiryOwner` names parent deadline/interaction timeout, child
interaction expiry/effective deadline, and outer tool timeout for cross-package waiting outcomes.

`HumanInteractionRequest.subagent_origin` defaults to `None`. Proxy requests carry a frozen
`SubagentProxyOrigin` containing only child run/interaction IDs, the agent selector, and expiry
owner. It round-trips through the existing request JSON without separate state or repeated catalog
membership validation.
Both `SubagentProxyOrigin` and `SubagentExpiryOwner` are imported from `iris.hitl`. The host uses
parent `resume()` for a current PENDING proxy; parent `recover()` automatically continues the exact
child from a RESOLVED response after a crash.

Field parsing first produces a complete typed request. `HumanInteraction` model-level validation
then compares the `tool_call_id`, request subject, and lifecycle delta without rechecking whether
required fields exist.

The standalone interaction store, consumed/resume phases, checkpoint payload, and stateful service
have been removed.

## Stateless service

`HumanInteractionService` constructs pending values and typed child proxies, validates responses
against exact run/interaction facts, expiry, and any stored response, and projects a response to `ToolResult` or
`ApprovedToolCall`. `project_response(interaction)` reads the RESOLVED/CLOSED interaction's response
without accepting a second copy. It performs no persistence. Harness uses lifecycle commands, including
`ResumeWaitingRun`, for atomic state transitions.

## Fingerprint

`make_call_fingerprint()` hashes canonical JSON for session/run/call/tool/arguments/workspace.
Approval applies to that exact subject only; mismatched call identity, arguments, or workspace rejects execution.

## Public API

`iris.hitl` exports the typed models, enums, fingerprint helper, and stateless service. It exports
no interaction store or compatibility adapter.

## Verification

```bash
uv run pytest tests/harness/test_runner_resume.py tests/runtime/test_execute.py tests/tools/test_executor_preflight.py tests/tools/test_human_ask_tool.py
uv run ruff check src/iris/hitl tests/harness/test_runner_resume.py tests/runtime/test_execute.py tests/tools/test_executor_preflight.py tests/tools/test_human_ask_tool.py
uv run mypy src/iris/hitl
```
