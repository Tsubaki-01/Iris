# iris.hitl

`iris.hitl` 定义持久化 human-in-the-loop 协议：人工权限确认与问题回答的请求、响应、
interaction 生命周期及其存储契约。

它不实现 UI、provider 调用、工具执行或普通 session 消息存储；宿主应用负责呈现
interaction，`iris.runtime.AgentRuntime` 负责创建、等待与恢复。`iris chat` 是当前首个
terminal host adapter，但 request、response 和 checkpoint 的权威模型仍在本包与 runtime。

## 核心模型

- `PermissionInteractionRequest`：一次精确工具调用的批准/拒绝请求，包含稳定
  `call_fingerprint`。
- `QuestionInteractionRequest`：`human.ask` 的单个问题与可选选项。
- `HumanInteraction`：持久化记录，含 request、response 与 JSON-safe checkpoint。
- `InteractionStatus`：`pending`、`resolved`、`consumed` 表示人工响应生命周期。
- `InteractionResumePhase`：`waiting`、`claimed`、`result_ready`、`result_committed`
  表示 runtime 恢复与结果提交进度；它不同于人工响应状态。

## 服务与存储

`HumanInteractionService` 在 `InteractionStore` 上执行 create、resolve、claim 和
`update_consumed` 的 CAS 状态转换。相同 resolved response 是幂等的，不同 response
会产生冲突。

`InMemoryInteractionStore` 适合测试和 `session.backend: none`，进程退出即丢失。
`SQLiteSessionStore` 同时实现 `InteractionStore`，将 interaction 保存到独立的
`human_interactions` 表，可跨 runtime 重建恢复。

## Runtime 边界

当 runtime 返回 `RuntimeStatus.WAITING_HUMAN` 时，调用方读取
`RuntimeTurnResult.pending_interaction` 并通过 host adapter 收集响应；随后调用
`await runtime.resume(interaction_id, response)`。

`iris chat` 当前支持：

- permission 使用 `[y/N]`；空输入表示 reject，approve 只覆盖展示的精确调用一次。
- `human.ask` 可按编号选择 option，也可输入自由文本。
- 同一个 run 连续出现多个 gate 时，adapter 按 runtime 返回顺序逐个恢复。
- Ctrl+C 或 EOF 只退出当前 adapter，不等于 reject/cancel，pending interaction 保持不变。
- SQLite backend 会在相同 session 下自动发现并恢复 interaction；内存 backend 不承诺跨进程。
- `claimed` 且执行结果未知时 fail closed，不自动重放工具。

权限策略先于 host 输入：`writes: confirm` 才创建写权限 interaction，`allow` 直接执行，
`deny` 不可由 CLI 批准覆盖。当前没有 TUI 或 Web adapter，也不提供超时、取消或长期授权规则。
