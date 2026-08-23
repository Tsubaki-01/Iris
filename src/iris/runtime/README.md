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
- 可选的 `RuntimeSteeringPort` 只向当前 activation 的安全边界提供瞬时输入，不拥有 queue 或
  persistence。

## 低层调用契约

```python
result = await runtime.execute(
    activation,
    commits=commit_port,
    cancellation=cancellation_signal,
    steering=steering_port,  # 可选；省略时保持原行为
)
```

`RuntimeActivationInput` 包含 run/activation/session identity、`start | resume | recover` kind、
固定的 `RuntimeExecutionOptions` 和 JSON-safe cursor。`RuntimeActivationResult` 只返回 engine
事实；调用方必须从 durable store 重载最终 `RunResult`。

`start` 与 `before_model / step 0` 的初始 `recover` activation 携带当前用户 input；`resume` 和
非初始 `recover` 不携带。engine 只在该字段存在时将其注入 provider request 一次，后续恢复依赖
committed session history。

cursor 位置只有：

- `before_model`：可预留下一次 provider step；
- `tool_batch`：provider response 已提交，按 `next_tool_index` 推进 exact tool calls；
- `outcome_ready`：assistant outcome 已提交，只差 lifecycle terminal settlement。

无工具的 provider response 会以 `CheckpointResumability.OUTCOME_READY` 提交。工具 effect 前
必须 durable claim，result 后必须 durable commit；claim 后无法证明结果时返回
`TOOL_OUTCOME_UNKNOWN`，不得重放 effect。

## Runtime steering

自定义 lifecycle owner 可以为一次 `execute()` 调用传入 activation-scoped
`RuntimeSteeringPort`。`claim(run_id, activation_id)` 每个安全边界最多返回一条
`SteeringInput`；该 frozen model 只包含非空 `submission_id` 与 `Role.USER` message。Runtime
不创建 queue，也不把 claim 状态写入 cursor、checkpoint 或 store。

Runtime 只在两个位置 claim：

- 无工具 assistant response 已生成、`commit_model_step` 之前；成功时将 assistant 与 steer
  user message 放入同一 delta，cursor 进入下一 `before_model`，resumability 为 `SAFE`；
- 同批次 final ordered tool result 已知、最终 `commit_tool_result` 之前；成功时将 tool result
  与 steer user message 放入同一 delta，沿用下一 `before_model` cursor。

中间工具结果、provider/tool effect 执行中、HITL waiting、`outcome_ready`、cancellation、deadline
和 STOP terminal error 都不会 claim。Claim 返回后到同步 commit 与 `acknowledge()` / `fail()`
之间没有 `await`：commit 成功才 acknowledge，commit 异常则 fail `commit_failed` 并原样传播；
callback 自身的异常只记录日志，不会覆盖 durable 结果。传入 `None` 或 claim 返回 `None` 时，
既有 cursor、message delta、resumability 与 outcome 语义不变。

## 有界工具并发

在 `RETURN_TO_MODEL` 策略下，runtime 会把连续的“只读且声明为并发安全”调用组成内部窗口，
每个窗口最多 8 条。8 是私有实现上限，不是 YAML、`RuntimeExecutionOptions` 或环境变量配置；
本次能力没有改变 public config、schema、model 或导出。

窗口只覆盖连续候选。STOP、HITL、preflight result、WRITE/EXECUTE/NETWORK/MCP/AGENT，以及任一
不安全或分类失败的调用都是串行屏障，后序调用不能跨过屏障启动。每个 child 在 body 前仍会
独立 revalidate 并提交 exact durable claim；body 可以乱序结束，但 result message、cursor、
session history、checkpoint 和 committed event 只按原始 ordinal 的连续前缀推进。多个
`TOOL_CALL_CLAIMED` telemetry event 的先后顺序不是契约。

control interruption 只提交首个异常/空洞之前的已知 `ToolResult`；后序内存结果不会跳洞。
任何未提交的 durable claim 都会让取消、deadline 或程序中断最终 fail closed 为
`OUTCOME_UNKNOWN`。父 task 或基础设施退出前，runtime 会 cancel 并 drain 自己创建的 children。
协作式取消使用 `iris.exceptions.IrisCancellationRequestedError`；runtime 将它转换为 activation
outcome，而不是普通工具错误。

并发文件读取共享同一个 `ReadFileState` identity；worker 只返回不可变 observation，由 event
loop 合并。窗口 settle 后的 checkpoint snapshot 包含合并记录，后续串行 write barrier 可以
继续执行 stale-read 检查。同步 callable 默认 inline；显式 `CallableExecutionMode.THREAD` 才把
阻塞 body 放入 worker。线程无法安全强停，取消或 timeout 只停止等待并丢弃晚到返回；claim 已
存在时 runtime 以 `OUTCOME_UNKNOWN` 收口，晚到结果不能推进 history、cursor 或 checkpoint。
thread placement 不承诺 CPU 加速。NETWORK/MCP 并发或 write 并发未来必须另行设计 effect、
retry、timeout、冲突与 crash reconciliation 协议，不能直接放宽当前 classifier；本轮也没有
引入 delta/merge/lock/hash 模型。

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

Factory 会先解析 `permissions.workspace`，再构建基础 context 和用户声明的工具。若
`skills.enabled: true`，它以该 workspace 做一次项目级发现快照：非空结果会追加
`available_skills` system slot，并在创建 `ToolRegistryView` / `ToolExecutor` 前注册共享同一
registry 的 `load_skill`。关闭 Skill 或发现结果为空时会精确绕过 catalog 和 loader，不改变
原有 context/tool 形状；每个 factory/runtime 实例内不自动刷新快照。

`skills.root` 越出 workspace、`skills.require` 缺失，或 `load_skill` 与用户工具名称/别名冲突，
都会在装配阶段转为 `IrisConfigError` 并 fail closed。完整契约见
[`iris.skill`](../skill/README.md)。

## 公开接口

包级导出包括 `AgentRuntime`、`RuntimeFactory`、`RuntimeEnvironment`、provider/assembler/tool
bridge、`RuntimeSteeringPort`、`SteeringInput`，以及 activation/commit-port contracts。不存在 complete-run options/status/result、
`run_turn()`、`run_loop()`、`resume()` 或旧 checkpoint helper。

## 验证

```bash
uv run pytest tests/runtime
uv run ruff check src/iris/runtime tests/runtime
uv run mypy src/iris/runtime
```
