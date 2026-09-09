[English](README.en.md)

# `iris.hitl`

`iris.hitl` 只定义 human-in-the-loop typed domain models 和无状态
`HumanInteractionService`。它不保存 interaction、不拥有 clock，也不执行工具；所有 durable
interaction facts 由同一个 `LifecycleStore` 与 run aggregate 一起提交。

## 领域模型

- `ToolCallSnapshot`：exact tool-call identity、arguments、workspace 与 SHA-256 fingerprint；
- `PermissionPrompt` / `QuestionPrompt`：两种人工请求；
- typed permission/question responses；
- `HumanInteractionRequest`：tool subject 与 prompt 信封；
- `HumanInteraction`：`pending | resolved | closed` 状态、version 与时间事实；
- `ApprovedToolCall`：批准后传给 engine 的 exact projection。

公开 `SubagentExpiryOwner` 统一命名 parent deadline/interaction timeout、child
interaction expiry/effective deadline 与 outer tool timeout，供跨包等待结果引用。

`HumanInteractionRequest.subagent_origin` 默认 `None`；proxy 请求保存 frozen
`SubagentProxyOrigin`，仅含 child run/interaction ID、agent selector 和 expiry owner。
它随既有 interaction request JSON 序列化，不另存状态或重验 catalog membership。
`SubagentProxyOrigin` 与 `SubagentExpiryOwner` 均从 `iris.hitl` 导入。Host 对当前 PENDING
proxy 使用 parent `resume()`；已 RESOLVED 的 crash gap 用 parent `recover()` 自动继续原 child。

字段 parsing 先产生完整 typed request；`HumanInteraction` 的 model-level 校验随后只比较
`tool_call_id`、request subject 和 lifecycle delta，不重复防御必填字段缺失。

旧 standalone store、consumed/resume phase、checkpoint payload 和 stateful service 已删除。

## 无状态服务

`HumanInteractionService` 提供以下无状态操作：

- `create_pending()`：从 active run snapshot 构造尚未持久化的 pending value；
- `create_subagent_proxy()`：从 typed child prompt 与 parent snapshot 构造 proxy；
- `validate_response()`：校验 run/interaction identity、kind、expiry 与 environment fingerprint；
- `project_response(interaction)`：读取 RESOLVED/CLOSED interaction 的存储回答，投影为
  `ToolResult` 或 `ApprovedToolCall`，不再接受第二份 response。

服务不做 persistence。Harness 通过 lifecycle `SuspendRun`、`ResolveInteraction`、
`ResumeWaitingRun` 和 `FinishRun` commands 完成原子状态转换。

## Fingerprint

`make_call_fingerprint()` 对 session/run/call/tool/arguments/workspace 的 canonical JSON 做 SHA-256。
批准只适用于该 exact subject；任何 identity 或环境漂移都必须 fail closed。

## 公开接口

`iris.hitl` 导出上述 typed models、enums、fingerprint helper 和无状态 service。不导出
interaction store 或兼容 adapter。

## 验证

```bash
uv run pytest tests/harness/test_runner_resume.py tests/runtime/test_execute.py tests/tools/test_executor_preflight.py tests/tools/test_human_ask_tool.py
uv run ruff check src/iris/hitl tests/harness/test_runner_resume.py tests/runtime/test_execute.py tests/tools/test_executor_preflight.py tests/tools/test_human_ask_tool.py
uv run mypy src/iris/hitl
```
