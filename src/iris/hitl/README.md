# iris.hitl

`iris.hitl` 定义持久化 human-in-the-loop 协议：人工权限确认与问题回答的请求、响应、
interaction 生命周期及其存储契约。

它不实现 UI、provider 调用、工具执行或普通 session 消息存储；宿主应用负责呈现
interaction，`iris.runtime.AgentRuntime` 负责创建、等待与恢复。

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
`await runtime.resume(interaction_id, response)`。没有内置 CLI、TUI 或 Web UI，也不提供
超时、取消或长期授权规则。
