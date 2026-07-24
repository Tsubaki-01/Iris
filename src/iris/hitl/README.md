# iris.hitl

`iris.hitl` 定义持久化 human-in-the-loop gate 协议：人工权限确认与问题回答共用请求信封、
interaction 生命周期及存储契约，同时保留两者不同的领域语义。

它不实现 UI、provider 调用、工具执行或普通 session 消息存储；宿主应用负责呈现
interaction，`iris.runtime.AgentRuntime` 负责创建、等待与恢复。`iris chat` 是当前首个
terminal host adapter，但 request、response 和 checkpoint 的权威模型仍在本包与 runtime。

## 核心模型

- `ToolCallSnapshot`：两类 gate 共用的精确工具调用身份，包含工具名、参数、workspace 与稳定
  `fingerprint`。
- `PermissionPrompt`：权限策略产生的批准/拒绝提示；批准只绑定当前 tool call snapshot。
- `QuestionPrompt`：`human.ask` 的单个问题与可选选项。
- `HumanInteractionRequest`：唯一的 `tool_call + typed prompt` 请求信封；旧 `subject` 字段
  不兼容。
- `HumanInteraction`：持久化记录，含 request、response 与 JSON-safe checkpoint。
- `InteractionStatus`：`pending`、`resolved`、`consumed` 表示人工响应生命周期。
- `InteractionResumePhase`：`waiting`、`claimed`、`result_ready`、`result_committed`
  表示 runtime 恢复与结果提交进度；它不同于人工响应状态。

## 服务与存储

`HumanInteractionService` 在 `InteractionStore` 上通过唯一 `create(request, ...)` 入口以及
resolve、claim 和 `update_consumed` 的 CAS 状态转换。相同 resolved response 是幂等的，
不同 response 会产生冲突。

`InMemoryInteractionStore` 位于 `src/iris/hitl/in_memory.py`，适合测试和
`session.backend: none`，进程退出即丢失。
`iris.store.SQLiteStore` 同时实现 `InteractionStore`，将 interaction 保存到独立的
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

权限策略先于 host 输入，`DENY` 具有最高优先级且不可由人工回答覆盖。human tool 仅在
`ALLOW` 时产生 question；若策略对 human tool 返回 `REQUIRE_HUMAN`，executor 会以
`PERMISSION_ERROR` fail closed，避免同一调用形成嵌套双 gate。普通工具只有在
`REQUIRE_HUMAN` 时产生 permission。当前没有 TUI 或 Web adapter，也不提供超时、取消或
长期授权规则。

模型可见的 `AskQuestionTool` 不属于 HITL 状态机，定义在
`iris.tools.builtin.human` 并由 `iris.tools` 顶层导出；本包只拥有 typed request/response、
checkpoint 生命周期和存储协议。
