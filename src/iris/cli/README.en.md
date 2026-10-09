[中文](README.md)

# `iris.cli`

`iris chat` is a terminal host for `AgentRunner` and one `SessionManager`. The main thread reads
input while a background event loop executes the Agent, consumes events, and handles human responses.
The CLI does not own authoritative run, Goal, or history state.

## Starting and ordinary input

```bash
iris chat agent.yaml --session-id work --max-steps 8
```

Models and tools follow [Agent configuration](../agents/README.en.md). `--env-file` selects a dotenv
file. `--max-steps` bounds model steps within each Run; `--no-tools` disables model tools and cannot
be used when creating an automatic Goal.

Startup first resolves `prompts.root` (default `.iris/prompts`) from the effective root workspace
and adds only missing named templates, then passes one `PromptSource` to Memory and the runner.
Existing project text is preserved. After manual edits, a new runner adopts Goal/Todo and
system/context guidance; compaction and automatic Memory adopt edits at the start of their next
complete operation or cycle. Existing `system` / `context` configuration stays in place;
see [project prompt sources](../prompts/README.md).

With `evolution.enabled` and `skills.enabled`, the CLI constructs a project experience service
from the main configuration and binds it to the host's shared coordinator. Memory and evolution
retain separate locks, cancellation state and cleanup. Evolution also works with Memory disabled.
Exit drains existing work without running an extra summary. A newly generated Skill is discovered
on the next runner startup; there is no additional maintenance command.

`prompt_targets/config_targets` permit finite revisions driven by specific issues in real tasks.
The CLI binds its primary YAML path to the candidate parser. Saved configuration takes effect after
restarting `iris chat`; a new session does not hot-switch an existing runner's configuration.

- Ordinary input starts a Run when idle or steers the current Run at an existing execution boundary.
- `/follow-up <message>` queues the next Run after the current one finishes.
- `/todo` reads the current session's checklist and displays its actual file path.
- Human responses use `y/yes/n/no` for permission, with empty input meaning reject; questions accept
  an answer or an option number.
- `/help` lists commands. `/exit`, `/quit`, and EOF exit. Ctrl-C cancels current execution and exits
  with code 130.

## Automatic maintenance

With `memory.generation.enabled`, the CLI binds the constructed Memory service to one shared
`MaintenanceCoordinator` and binds the runner before accepting input. `maintenance.idle_seconds`
sets the quiet interval (300 seconds by default). A new automatic source batch also requires
`maintenance.min_pending_runs` eligible new Runs, defaulting to 10. Set it to 1 for earlier processing
after eligible material appears; the idle interval still applies. Generation budgets remain under
`memory.generation`. Continuation of admitted batches, projection repair, and publication settlement
do not wait for another batch of new Runs.

Automatic learning consumes only terminal, fully captured Runs. A session waiting for a human response
keeps its materials pending. New foreground work cancels uncommitted generation. On exit, the host
stops SessionManager, drains actual maintenance IO, then closes runner-owned resources and the event
loop. Shutdown never starts extra model generation.

With the Agent's `observability.enabled`, the CLI creates one shared service using the global
`observability` export configuration and passes it to the runner, Memory, Evolution, and coordinator.
See [observability](../observability/README.md) for installation and the complete OTLP endpoint.
Content capture is off by default; disabled observation creates no SDK or exporter. The CLI closes
its service after business resources finish, including cleanup after construction or preparation
failure. A direct `run_chat_loop(runner=...)` call does not own the runner's borrowed observation
service; explicitly passing `observability=...` transfers its close responsibility to the chat host.

## Viewing a Todo checklist

Set `todo.enabled: true` in the Agent configuration and keep `context_policy.enabled: true`, then
rebuild the Agent. `/todo` displays the completed/total count, absolute file path, and every item:
`[ ]` for pending, `[-]` for in progress, and `[x]` for completed.

A missing or empty checklist displays `暂无待办` (no tasks); an invalid file displays its path and
diagnostic instead of a completion count. After a manual edit, run `/todo` again to read the latest
content. Viewing does not create a file or remove completed items. Configure ordinary file tools
explicitly if the model needs to maintain the file; see [Todo](../todo/README.md).

Viewing also works during ordinary or Goal execution and while a question or permission response
is pending. `/todo` never becomes chat, steering input, or an interaction answer; the next actual
answer still resumes the original interaction. The command accepts no arguments: `/todo extra`
shows `用法：/todo` (usage). Disabled Todo and file read failures display an error while chat continues.
The CLI does not enable Todo automatically or poll for file changes.

## Automatic Goals

Add the following to an existing Agent configuration:

```yaml
context_policy:
  enabled: true
goal:
  enabled: true
  max_rounds: 20
session:
  backend: sqlite
```

Then provide an explicit objective and acceptance condition:

```text
/goal Fix empty-input handling in the calculation function and pass its specified tests
```

A Goal may span several Runs. Each Run already supports multiple model and tool calls; work is not
split into fixed stages. The CLI waits only for the control receipt, so input stays available.
User input and queued follow-ups take precedence over automatic continuation.

| Command | Behavior |
| --- | --- |
| `/goal` | Show usage. |
| `/goal <objective>` | Create and arm a Goal using the CLI's current model-step and tool options. |
| `/goal status` | Read objective, status, armed state, rounds, reason, occupied Run, interaction, and error. |
| `/goal edit <objective>` | Replace the objective and pause, retaining the Goal ID and spent rounds. |
| `/goal edit --max-rounds 30` | Change the total allowance and pause, retaining text and spent rounds. |
| `/goal pause` | Stop subsequent rounds while the current Run may finish. |
| `/goal resume` | Explicitly resume, reusing existing execution or the original human interaction. |
| `/goal complete` | Declare completion as the user, allowing the current Run to finish. |
| `/goal clear` | Deselect the current Goal while preserving historical Goals and Run bindings. |

Escape an objective beginning with a reserved word using `/goal -- status analysis requirements`.
To edit text beginning with an option prefix, use `/goal edit -- --max-rounds is text`.
Internal spaces and quotation marks remain intact. Invalid arguments show usage and never become
ordinary chat or steering input. Disabled Goal only produces a configuration hint; the CLI does not
edit YAML.

Rounds are spent at Run admission and are not refunded; resume does not reset them. The final allowed
round can still complete a Goal. After exhaustion, raise the total with `edit --max-rounds` and then
`resume`. Completion comes from a user declaration or a committed model report, not independent
acceptance testing. Use Ctrl-C to stop current execution immediately.

A new manager does not automatically resume old Goals. WAITING continues the original question.
For an existing ACTIVE Run without a live invocation, `resume` displays its run/activation ID and
points to the explicit SDK call `await manager.goal.resume(expected_activation_id="...")`.
The CLI has no general lifecycle recovery command. SQLite uses the new schema contract without old
schema migration; children and forks do not inherit Goals. See [goal](../goal/README.md) for
control receipts, absolute deadlines, and recovery rules.

## Implementation and verification

`_ChatSessionHost` in `chat.py` dispatches all Goal controls on the manager's event loop. GoalChanged
shares the mixed stream with Run events and displays the latest Goal snapshot after relevant terminal
events. Live text output uses the same GoalView formatting without a second duplicate Goal output
path. Notifications can coalesce; use `status` to read current facts. Command cleanup and Goal storage
failures display actual errors rather than fabricated completion.

`tests/cli/test_chat_goal.py` uses the real host, manager, runtime, and store with a controlled provider
to cover two-round completion, command dispatch, text preservation, disabled Goal, and explicit recovery
hints. These tests do not evaluate a real model's task quality.

`tests/cli/test_chat_todo.py` exercises the real `run_chat_loop` for all three task states, manual
edits, query failures, argument usage, and read-only behavior during ordinary/Goal execution and
question/permission WAITING states.

Guides and reference (Chinese): [Quick start](../../../docs/getting-started/quickstart.md) · [CLI reference](../../../docs/reference/cli.md).
