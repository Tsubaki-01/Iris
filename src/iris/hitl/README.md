[English](README.en.md)

# `iris.hitl`

`iris.hitl` 定义 human-in-the-loop gate 的 JSON-safe 领域模型和无状态投影服务。权限
确认和人工问答共享精确的 tool-call identity，但保留不同的 prompt/response 语义。

本包不实现 UI、provider 调用、工具执行或持久化事务。新 lifecycle 路径中，
`iris.lifecycle.LifecycleStore` 是 run、checkpoint、tool call 和 interaction 写入的唯一权威；
`HumanInteractionService` 只构造、校验和投影值。

## 核心模型

- `ToolCallSnapshot`：工具 ID、名称、参数、workspace 和稳定 SHA-256 `fingerprint`。
- `PermissionPrompt` / `PermissionInteractionResponse`：精确一次工具权限的 approve/reject。
- `QuestionPrompt` / `QuestionInteractionResponse`：一个问题、可选选项和非空回答。
- `HumanInteractionRequest`：唯一的 `tool_call + typed prompt` 信封。
- `HumanInteraction`：绑定 run/session/step/tool call 的 durable fact，状态为
  `pending -> resolved -> closed`，并可声明 aware `expires_at`。
- `ApprovedToolCall`：权限批准的 frozen 投影 DTO；它不授权直接执行副作用。

`consumed` 状态和 `InteractionResumePhase` 仍为 Phase 5 删除前的旧 runtime 表征路径保留，
不属于新 `AgentRunner` 的 interaction 流程。

## 无状态服务

```python
from iris.hitl import HumanInteractionService

service = HumanInteractionService()
```

`HumanInteractionService` 是零参构造，不持有 store 或 clock，提供：

- `create_pending(request, *, run, step_index, expires_at)`：从 active run snapshot 构造未持久化
  interaction；
- `validate_response(interaction, *, run, response, now, environment_fingerprint)`：校验
  waiting identity、response kind、expiry 和 environment fingerprint；
- `project_response(interaction, response)`：把已解决的 question 投影为答案
  `ToolResult`，reject 投影为 `USER_REJECTED` 结果，approve 投影为
  `ApprovedToolCall`。

所有方法都无持久化副作用。批准后 runtime 仍必须根据当前环境重新鉴权，并在工具
effect 前通过 lifecycle store claim 精确 prepared call。

## Resume 执行链

Host 展示 `RunResult.pending_interaction` 并收集 typed response，然后调用：

```python
result = await runner.resume(
    run_id,
    interaction_id=interaction.interaction_id,
    response=response,
)
```

`AgentRunner.resume()` 先加载 waiting run 和精确 interaction，惰性结算已到期的 run deadline/
interaction expiry，再调用无状态服务校验 response。Lifecycle store 以 CAS 持久化
resolved response，关闭旧 interaction，并为同一 `run_id` 创建新 activation。之后从
checkpoint v1 恢复同一 engine。纯 read 不会隐式结算 expiry。

相同 response 的 durable resolve 可幂等重试；不同 response、错误 run/interaction/kind/version/
fingerprint 或环境漂移均 fail closed。Same-batch 中的后续 gate 按原顺序逐个暴露，
一个 run 同时最多一个 open interaction。

## 公开接口与维护

`iris.hitl` 顶层导出 typed prompts/responses、`HumanInteraction`、`ApprovedToolCall`、
`HumanInteractionService` 和 `make_call_fingerprint()`。`InteractionStore` 与
`InMemoryInteractionStore` 目前仍为旧 runtime 兼容而导出，新代码应使用 `iris.lifecycle.LifecycleStore`，
不要将两个存储路径组合或双写。

| 修改内容 | 主要位置 | 对应测试 |
| --- | --- | --- |
| JSON-safe 模型与状态约束 | `models.py` | `tests/hitl/test_models.py` |
| create/validate/project 语义 | `service.py` | `tests/hitl/test_service.py` |
| aggregate interaction 事务 | `../store/in_memory.py`, `../store/sqlite.py` | lifecycle store contract |
| durable resume 与 expiry | `../harness/runner.py` | runner resume/expiry tests |

```bash
uv run pytest tests/hitl/test_service.py tests/harness/test_runner_resume.py tests/harness/test_runner_interaction_expiry.py
uv run ruff check src/iris/hitl tests/hitl
uv run mypy src/iris/hitl
```
