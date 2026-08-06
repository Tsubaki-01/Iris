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

旧 standalone store、consumed/resume phase、checkpoint payload 和 stateful service 已删除。

## 无状态服务

`HumanInteractionService` 只有三个职责：

- `create_pending()`：从 active run snapshot 构造尚未持久化的 pending value；
- `validate_response()`：校验 run/interaction identity、kind、expiry 与 environment fingerprint；
- `project_response()`：把回答投影为 `ToolResult`，或把批准投影为 `ApprovedToolCall`。

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
uv run pytest tests/hitl tests/harness/test_runner_resume.py
uv run ruff check src/iris/hitl tests/hitl
uv run mypy src/iris/hitl
```
