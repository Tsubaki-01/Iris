[中文](README.md)

# Iris Demo

This directory is a runnable CLI/runtime integration example. It demonstrates Iris's config-first
path: `agent.yaml` selects the model, structured context, tools, workspace, permissions, and SQLite
session; `iris chat` hosts a multi-turn terminal session.

## Contents

- `agent.yaml`: DeepSeek `deepseek-chat`, `context.yaml`, file and human tools, and SQLite session.
- `context.yaml`: system, memory, and before-current-input context sections.
- `workspace/`: the only filesystem area exposed to the demo agent.
- `.iris/demo-session.db`: generated session and HITL persistence.
- `trace.jsonl`: optional generated provider request/response trace.
- `lifecycle-agent.yaml`: dedicated DeepSeek + SQLite configuration for `iris run`.
- [`LIFECYCLE_SMOKE.en.md`](LIFECYCLE_SMOKE.en.md): manual acceptance with explicit lifecycle identities.

## Run

Use Python `>=3.12`, install the repository dependencies with `uv`, and configure a DeepSeek key:

```bash
export IRIS_PROVIDER_API_KEYS__DEEPSEEK=sk-xxx
uv run iris chat demo/agent.yaml \
  --session-id demo \
  --trace compact \
  --trace-file demo/trace.jsonl
```

You may instead pass a dotenv file explicitly:

```bash
uv run iris chat demo/agent.yaml \
  --env-file .env.local \
  --session-id demo \
  --trace compact \
  --trace-file demo/trace.jsonl
```

Useful commands are `/help`, `/trace off|compact|full`, and `/exit` or `/quit`. `--no-tools`
exercises the conversation path without exposing tool schemas.

## Tools, permissions, and explicit lifecycle operations

The demo enables `file.read`, `file.list`, `file.grep`, `file.write`, `file.edit`, and
`human.ask` (model-visible name `ask_question`). File access remains inside `demo/workspace/`.
`permissions.writes: confirm` creates a permission gate before every write:

- `y` or `yes` approves only the exact displayed call once.
- empty input, `n`, or `no` rejects it without changing the target.
- other input is rejected by the host adapter and does not call `resume()`.

Questions accept a one-based option number or free text. Multiple gates from one assistant response
are resumed in runtime order before the terminal result is rendered.

`iris chat` owns only its current interactive process and does not guess or recover a run by session
after restart. Ctrl+C/EOF does not mean reject, cancel, or recover. For cross-process durable run
operations, use `iris run start/status/events/resume/cancel/recover` and pass the run, interaction,
and activation identities explicitly. `events` performs one provider-free durable read for an
explicit run ID and exclusive `--after-sequence` cursor; it does not watch, discover runs, or mutate
lifecycle state.

See [`LIFECYCLE_SMOKE.en.md`](LIFECYCLE_SMOKE.en.md) for the real DeepSeek + SQLite three-scenario
acceptance flow. It uses the separate `lifecycle-agent.yaml` and `.iris/lifecycle-smoke.db`, leaving
the ordinary chat demo tool surface and database unchanged.

## Generated data and maintenance

Session data goes to `demo/.iris/demo-session.db`; trace output is appended to
`demo/trace.jsonl`; tool effects must stay inside `demo/workspace/`. This is an integration example,
not a separate Python package.

```bash
uv run pytest tests/cli tests/context tests/agents
```
