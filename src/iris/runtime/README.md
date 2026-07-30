[English](README.en.md)

# `iris.runtime`

`iris.runtime` 是 Agent lifecycle 的低层 inner engine。它从一个已持久化的
`RuntimeCursor` 开始，通过调用方提供的 `RuntimeCommitPort` 推进 provider 与工具循环，直到
completed、waiting、budget、cancel、deadline、failed 或 outcome unknown。它不创建 logical
run、不选择 store，也不拥有 cancellation/recovery 的公开编排。

完整运行请使用 `iris.harness.AgentRunner`。只有实现自定义 lifecycle owner 时才直接调用
`AgentRuntime.execute()`。

## 依赖方向

```text
AgentRunner -> AgentRuntime.execute -> RuntimeCommitPort
     |                                  |
     +------------ LifecycleStore <-----+
```

- `RuntimeFactory` 只装配 context、provider、tools、workspace 与可选 memory service；
- `RuntimeEnvironment` 只保存 engine live dependencies，没有 session/lifecycle store 或
  interaction service；
- runtime 不 import harness，也不直接写 SQLite；
- exact session、checkpoint、tool claim/result 与 interaction 写入由 commit port 提供。

## 低层调用契约

```python
result = await runtime.execute(
    activation,
    commits=commit_port,
    cancellation=cancellation_signal,
)
```

`RuntimeActivationInput` 包含 run/activation/session identity、`start | resume | recover` kind、
固定的 `RuntimeExecutionOptions` 和 JSON-safe cursor。`RuntimeActivationResult` 只返回 engine
事实；调用方必须从 durable store 重载最终 `RunResult`。

cursor 位置只有：

- `before_model`：可预留下一次 provider step；
- `tool_batch`：provider response 已提交，按 `next_tool_index` 推进 exact tool calls；
- `outcome_ready`：assistant outcome 已提交，只差 lifecycle terminal settlement。

无工具的 provider response 会以 `CheckpointResumability.OUTCOME_READY` 提交。工具 effect 前
必须 durable claim，result 后必须 durable commit；claim 后无法证明结果时返回
`TOOL_OUTCOME_UNKNOWN`，不得重放 effect。

## 显式 Memory 注入

`RuntimeExecutionOptions.memory_query` 和 `memory_results` 是显式 opt-in 的动态 memory 输入。
每个 logical run 只在第一次 `before_model` step 注入一次；同一用户输入后续因工具循环或
HITL resume 产生的 provider 请求不会再次附加这条动态 memory。新的用户输入会创建新的
`start` activation，因此可以重新注入一次。`context.yaml` 中声明的静态 memory slot 不受此
规则影响。

## Factory

```python
from iris.runtime import RuntimeFactory

runtime = RuntimeFactory.from_config_path("agent.yaml", provider=provider)
```

Factory 不读取或创建 lifecycle database。`agent.yaml` 的 `session` 配置由 harness composition
解释；直接调用 Factory 时该字段不会产生持久化副作用。

## 公开接口

包级导出包括 `AgentRuntime`、`RuntimeFactory`、`RuntimeEnvironment`、provider/assembler/tool
bridge，以及 activation/commit-port contracts。不存在 complete-run options/status/result、
`run_turn()`、`run_loop()`、`resume()` 或旧 checkpoint helper。

## 验证

```bash
uv run pytest tests/runtime
uv run ruff check src/iris/runtime tests/runtime
uv run mypy src/iris/runtime
```
