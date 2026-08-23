[English](README.en.md)

# `iris.lifecycle`

`iris.lifecycle` 是 logical run 的纯数据与同步 store contract。它定义不可变 run/session/
activation/checkpoint/tool-call/event/result 模型、JSON-safe validation、投影函数和 CAS commands，
但不拥有运行控制流或具体数据库。

## 依赖边界

```text
harness -> lifecycle <- store
runtime  -> lifecycle
```

Lifecycle 不 import `iris.harness`、`iris.runtime` 或 `iris.store`。`AgentRunner` 是 owner，
`InMemoryLifecycleStore`/`SQLiteStore` 是实现，`AgentRuntime` 只消费 options/error contracts。

## Aggregate 不变量

- 一个 session 同时最多一个 non-terminal run lane；
- active run 恰好一个 current activation fence；waiting run 恰好一个 open interaction；
- model step 先 reserve 再 commit，最多一个未提交 reservation；
- tool effect 先 claim，再 commit result；unresolved claim 不得重放；
- terminal run 没有 current activation、open interaction 或 lane；
- terminal run 的 durable history 中，每个 `tool_use` 都恰好有一个匹配的 result；tool-call phase
  继续区分已提交结果、结果未知与从未开始，不能用合成 closer 抹去副作用知识；
- run、checkpoint、session revision、usage counters 与 environment fingerprint 必须交叉一致；
- mutation events 与 aggregate facts 同事务追加，sequence 单调递增。

## Checkpoint v1

`RunCheckpoint.resumability` 只有：

- `safe`：可以从 cursor 重新进入 engine；
- `outcome_ready`：assistant outcome 已提交，只补 terminal；
- `blocked_unknown`：effect 结果不可安全解释，禁止自动执行。

checkpoint schema 不迁移旧 payload，也不保存 provider client、task、lock、signal 或 callback。

## Store contract

`LifecycleStore` 提供 create/begin/reserve/model commit/tool claim/tool result/suspend/resolve/
cancellation/finish/recover commands，以及 run/session/lane/interaction/checkpoint/tool/result/event
reads。
每个 mutation command 携带 expected revision/fence；stale writer 必须 conflict，而不是覆盖新事实。
`load_session_lane()` 只是 lane owner 的只读发现入口，不承担恢复、修补或 ownership transfer。
`load_tool_call(run_id, tool_call_id)` 按 exact composite identity 返回单条 tool fact；
`load_run_control(run_id)` 只返回 `RunControlSnapshot` 的八个 fence/cancellation 字段。两者都不
替代 mutation CAS，也不改变同步 store boundary。

## 公开接口

`iris.lifecycle` 可直接导入所有契约模型、enums、commands、`LifecycleStore` 和
`snapshot_run()`/`project_result()`，包括最小只读投影 `RunControlSnapshot`。完整运行入口只在
`iris.harness`。

## 验证

```bash
uv run pytest tests/store tests/harness
uv run ruff check src/iris/lifecycle
uv run mypy src/iris/lifecycle
```
