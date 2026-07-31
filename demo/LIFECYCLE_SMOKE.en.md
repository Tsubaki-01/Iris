[中文](LIFECYCLE_SMOKE.md)

# `iris run` Lifecycle Manual Smoke

This guide validates `start/status/events/resume/cancel/recover` from the repository root against a
dedicated SQLite database. Events is a provider-free read; the other scenarios use the real DeepSeek
`deepseek-chat` service, network access, and provider quota, so do not put them in pytest or CI. The
three scenarios use separate runs. Cancel and
recover are alternative branches and must not be applied sequentially to one already-terminal run.

## Setup and reset

Set the key and local uv cache in every new PowerShell window:

```powershell
$env:IRIS_PROVIDER_API_KEYS__DEEPSEEK = "sk-xxx"
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
```

You may instead add `--env-file .env.local` to later commands. The reset below is destructive but is
limited to the dedicated smoke database and its SQLite sidecars; it does not touch
`demo/.iris/demo-session.db`. Close all smoke processes first:

```powershell
Remove-Item -LiteralPath @(
  "demo\.iris\lifecycle-smoke.db",
  "demo\.iris\lifecycle-smoke.db-wal",
  "demo\.iris\lifecycle-smoke.db-shm"
) -Force -ErrorAction SilentlyContinue
```

## Scenario A: HITL waiting, restart, and resume

Start a run with fixed identities:

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run start demo/lifecycle-agent.yaml --input "Call ask_question exactly once with question 'Choose smoke environment' and options ['test', 'production']. Do not answer it yourself." --session-id lifecycle-hitl --run-id lifecycle-hitl-001 --json
```

The start command must exit `0` with `ok=true`, `run.phase="waiting"`, and
`run.run_id="lifecycle-hitl-001"`. Save `pending_interaction.interaction_id`; the command below calls
it `INTERACTION_ID`.

The original process has exited. Submit the typed question answer from a new PowerShell process:

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run resume demo/lifecycle-agent.yaml --run-id lifecycle-hitl-001 --interaction-id INTERACTION_ID --answer "test" --json
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run status demo/lifecycle-agent.yaml --run-id lifecycle-hitl-001 --json
```

Resume and status must both exit `0`, return the same run ID, and end with
`run.phase="terminal"`, `run.stop_reason="completed"`, and an assistant message. If the model does
not call `ask_question`, reset the dedicated DB and retry with the explicit “exactly once / do not
answer” wording. Do not fabricate an interaction ID.

## Scenario B: cross-process cancel in two terminals

Start an independent run in terminal A and keep it open after the marker appears:

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run start demo/lifecycle-agent.yaml --input "Call wait_for_seconds exactly once with seconds=30, then report its result." --session-id lifecycle-cancel --run-id lifecycle-cancel-001 --json
```

Expected marker:

```text
IRIS_LIFECYCLE_SMOKE_TOOL_STARTED seconds=30
```

After the marker, run these commands in terminal B:

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run cancel demo/lifecycle-agent.yaml --run-id lifecycle-cancel-001 --reason "phase 6 cross-process smoke" --settlement-timeout 45 --json
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run status demo/lifecycle-agent.yaml --run-id lifecycle-cancel-001 --json
```

The synchronous tool is intentionally non-cooperative. Cancel must wait for the safe check after the
tool returns and must not report early settlement. Both final outputs must exit `0` with
`run.phase="terminal"` and `run.stop_reason="cancelled"`.

## Scenario C: claimed effect, interruption, and fenced recover

Start the third independent run in terminal A:

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run start demo/lifecycle-agent.yaml --input "Call wait_for_seconds exactly once with seconds=60, then report its result." --session-id lifecycle-recover --run-id lifecycle-recover-001 --json
```

Immediately press Ctrl+C in terminal A after this marker appears; do not run cancel:

```text
IRIS_LIFECYCLE_SMOKE_TOOL_STARTED seconds=60
```

Read the durable active snapshot from a new process:

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run status demo/lifecycle-agent.yaml --run-id lifecycle-recover-001 --json
```

Status must exit `0` with `run.phase="active"`. Save the exact `run.current_activation_id` as
`ACTIVATION_ID`; do not guess or substitute a newer ID. Then run:

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run recover demo/lifecycle-agent.yaml --run-id lifecycle-recover-001 --activation-id ACTIVATION_ID --json
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run status demo/lifecycle-agent.yaml --run-id lifecycle-recover-001 --json
```

Recover and final status must both exit `1` with `ok=false`, `run.phase="terminal"`,
`run.stop_reason="outcome_unknown"`, and the durable tool error. No second wait-tool marker may
appear during recovery; the claimed effect was not replayed.

## Common check: durable event timeline

After any scenario creates a run, inspect its complete durable event timeline from a new PowerShell
process with this read-only command:

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run events demo/lifecycle-agent.yaml --run-id lifecycle-hitl-001 --after-sequence 0 --json
```

The command must exit `0`. Events preserve the store's ascending sequence order, and every sequence
is strictly greater than the input `after_sequence`. Use the response's `next_after_sequence` as the
next cursor. If no new events exist, the response has `events=[]` and keeps the input cursor. Each
invocation reads once: it does not watch, call the provider, or mutate the run, and needs no DeepSeek
key.

## Output contract and troubleshooting

- With `--json`, a durable result writes one compact JSON object to stdout; an operation error without
  a durable run writes to stderr.
- The wait-tool marker is written to stderr, so scriptable JSON stdout remains unpolluted.
- Exit `0` means waiting, active, or a non-failure terminal outcome; exit `1` means an operation error,
  failed, or outcome_unknown; argparse uses `2`; Ctrl+C uses `130`.
- Status, events, and cancel do not require a DeepSeek key. Start, resume, and recover require a real
  key.
- `RUN_NOT_FOUND` usually means the run ID, config path, or dedicated DB differs between commands.
- `RUN_CONFLICT` means an interaction or activation identity is stale or mismatched. Do not retry with
  an automatically selected identity.
- If the provider does not select the requested tool, reset the dedicated DB, use a new fixed run ID,
  and make the request more explicit. A plain assistant response is not a pass.
- To keep human next commands directly copyable in PowerShell, the custom run/session identities in
  this guide use only ASCII letters, digits, and hyphens. Do not put quotes or PowerShell metacharacters
  in an identity.

## Acceptance record

| Item | Record |
| --- | --- |
| Date | |
| provider/model | DeepSeek / `deepseek-chat` |
| Scenario A run ID / phase / stop reason / PASS-FAIL | |
| Scenario B run ID / phase / stop reason / PASS-FAIL | |
| Scenario C run ID / phase / stop reason / PASS-FAIL | |
| Interaction ID | |
| Activation ID | |
