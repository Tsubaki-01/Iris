[English](README.en.md)

# `iris.lifecycle`

公开 Sub Agent 契约以 `SubagentRunLink(parent_run_id, parent_tool_call_id, child_run_id)`
关联独立运行；parent 工具保持 PREPARED。Store 增加 `AdmitChildRun`、`RebindSubagentProxy`、
`FinalizeSubagentResult` 与 exact link point read。Rebind 返回完整 WAITING `RunCommit`；
WAITING finalize 在同一 commit 返回绑定 fresh RESUME activation 的 ACTIVE run/checkpoint，
调用方无需追加普通 resume mutation。Child usage 保持 run-local。
四个类型直接从 `iris.lifecycle` 导入。Rebind 只更新 proxy binding 与 checkpoint sequence，
保持 history/usage/cursor；Finalize 才提交 parent result/message/usage/cursor。三条 mutation
沿用既有 revision、activation fence 和 store 事务，不增加新的并发字段。

`iris.lifecycle` 是 logical run 的纯数据与同步 store contract。它定义不可变 run/session/
activation/checkpoint/tool-call/event/result 边界模型、JSON-safe validation、投影函数和 CAS
commands，但不拥有运行控制流或具体数据库。持久化模型使用 Pydantic 校验 raw/load 数据；
同进程 command 使用 frozen slots dataclass，只携带调用方已经类型化的事实。

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

checkpoint 只接受当前 payload 形状，也不保存 provider client、task、lock、signal 或 callback。

## Store contract

`LifecycleStore` 提供 create/begin/reserve/model commit/tool claim/tool result/suspend/resolve/
cancellation/finish/recover commands，以及 run/session/lane/interaction/checkpoint/tool/result/event
reads。
`RunCommit.session_revision` 只在 mutation 改变 history 时返回提交后的 revision，不携带完整
`SessionSnapshot`；需要 history 时显式调用 `load_session()`。精确重试返回当前事实与空 events，
而不是第一次提交的旧快照。
每个 mutation command 携带 expected revision/fence；stale writer 必须 conflict，而不是覆盖新事实。
Store 只校验当前 mutation 影响的 phase、counter、identity 与 fence，再应用 typed delta；不会为了
更新单个字段而把整个已验证 aggregate `model_dump()` 后重新 `model_validate()`。SQLite row 与
checkpoint recovery 仍是完整验证边界，JSON-safe 约束仍由 durable model/encoder 保证。
`SessionSnapshot` 继续只公开 `session_id`、CAS `revision` 和完整 `messages`。revision 每次非空
message delta 只推进一次，与 delta 中的消息条数无关；持久化层的 message count 与 ordinal
不进入 lifecycle 公共模型或 command。
`load_session_lane()` 只是 lane owner 的只读发现入口，不承担恢复、修补或 ownership transfer。
`load_tool_call(run_id, tool_call_id)` 按 exact composite identity 返回单条 tool fact；
`list_tool_calls(run_id, step_index=...)` 将有序工具读取限定为一个模型步，省略时返回整 run；
`load_run_control(run_id)` 只返回 `RunControlSnapshot` 的 session 归属与 fence/cancellation 字段。
gateway 可以据此确认 run 属于所请求的 session，无需加载完整 run。上述读取都不
替代 mutation CAS，也不改变同步 store boundary。

## 公开接口

`CreateRun` 的 request/checkpoint identity 及 `RecoverActiveRun` 的 disposition/activation 组合
由公开 command 构造时校验；store 不重复检查这些已确定的 optional 关系，只检查当前 durable
事实及 mutation 所需的 CAS、identity 与 fence。

`iris.lifecycle` 可直接导入所有契约模型、enums、commands、`LifecycleStore` 和
`snapshot_run()`/`project_result()`，包括最小只读投影 `RunControlSnapshot`。完整运行入口只在
`iris.harness`。

## 验证

```bash
uv run pytest tests/store tests/harness
uv run ruff check src/iris/lifecycle
uv run mypy src/iris/lifecycle
```
