# Iris 的 Inspect AI 接入

本目录提供仓库级 `iris_solver()`：让 Inspect 调用真实 `AgentRunner`，取得输出、
运行状态和已有用量。该 Inspect 接入目前没有题集、benchmark 适配器或评分器。
本目录不随 `iris` Python 包发布，使用时从仓库根目录执行。

## 安装与配置

在仓库根目录安装独立的评测依赖组。它包含 Inspect AI，以及用于取消收尾的 AnyIO：

```shell
uv sync --group eval
```

普通 Iris 用户无需安装这个依赖组。主模型、工具、工作区和 memory 仍由 Agent YAML 配置；
provider 凭据继续使用 `iris.config`。下面只创建接口，不发起模型请求：

```python
from evals.iris_solver import iris_solver
from iris.config import init_config

init_config()
solve = iris_solver("examples/chat/agent.yaml")
```

未来将 `solve` 作为 Inspect `Task` 的 `solver`。Inspect 外层可使用 `mockllm/model` 作为
占位模型；真正被测模型由 Iris YAML 决定。接入层不会调用 Inspect 的 `generate()`，不会
使用 Agent Bridge 改写 Iris 的模型请求，也不会为了收集结果启用 streaming。

## 默认执行与扩展接口

`iris_solver(config_path, *, execute=None)` 返回 Inspect `Solver`。配置路径在构造时解析；
每次 sample 调用都会创建独立 runner、随机 session ID 和 `InMemoryLifecycleStore`。
因此本接入不使用 YAML 的 session SQLite 后端。

默认读取 **当前 `state.messages` 中唯一一条纯文本用户消息**，支持前置 solver 改写该文本。
system prompt 来自 Iris YAML。多条消息、附加 system 消息或多模态内容会明确报错，
由具体任务的 `execute` 回调决定如何映射，不静默丢弃内容。

自定义回调的签名是 `async execute(sample: IrisSample, state: TaskState) -> RunResult`。
它拿到完整 Inspect 状态，但接入层不会自动把题目 metadata、参考答案或评分配置发送给 Iris。

| 接口 | 用途 |
| --- | --- |
| `await sample.start(text, options=...)` | 在本 sample 的 session 中开始一轮；直接传递 `AgentRunOptions` |
| `await sample.resume(waiting_result, response)` | 用具体任务提供的 typed HITL response 恢复同一个 run |
| `sample.results()` | 按创建顺序读取已有结果，同一个 run 只保留最新累计状态 |
| `sample.runner` | 用 SDK 查询历史、工具记录等任务产物 |
| `sample.session_id` | 本 sample 的 session 标识 |

例如下面只定义带模型步数限制的执行接口，同样不会立即调用模型：

```python
from inspect_ai.solver import TaskState

from evals.iris_solver import IrisSample, iris_solver
from iris.lifecycle import AgentRunOptions, RunLimits, RunResult


async def execute(sample: IrisSample, state: TaskState) -> RunResult:
    return await sample.start(
        state.user_prompt.text,
        options=AgentRunOptions(limits=RunLimits(max_model_steps=8)),
    )


solve = iris_solver("examples/chat/agent.yaml", execute=execute)
```

任务回调可顺序执行多轮 start/resume，返回作为最终输出的那份 `RunResult`。
run 的创建与恢复使用这两个方法；查询使用 `sample.runner`。回调不另开后台运行任务，
不关闭或跨 sample 复用 runner。涉及真实 benchmark 时，再由该 benchmark 的协议决定
WAITING 回答、多轮输入以及任务结束条件；本接口不会自动批准权限或编造回答。

session/store 隔离**不等于文件、memory 或外部服务隔离**：这些仍使用 YAML 中的配置。
当前接口没有容器环境管理；具体题目环境及其清理由后续任务接入负责。

## 返回结果与资源生命周期

最终回答映射到 `state.output.completion`；有 assistant 消息时，将这条最终输出追加到
Inspect 消息列表。完整 Iris 轨迹不自动转换成 Inspect 消息历史。
`state.completed=True` 只表示 solver 已完成执行，**不是题目评分成功**。

`state.store.get("iris")` 读取接入层保留的结果命名空间，包含：

- `session_id`：本 sample 的 session。
- `output_run_id`：回调选定的最终输出所属 run。
- `runs`：本 sample 通过 `start()` 创建的 run，按创建顺序列出；每项含 `run_id`、
  `phase`、原始 `stop_reason`、`error`、`pending_interaction` 和原始 `usage`。

结果中不添加 `success` 或分数。FAILED、budget exhausted 和 WAITING 都保留各自语义。
默认遇到 WAITING 会记录待回答 interaction，然后在资源收尾时取消该 run；记录保留
**收尾之前**的 WAITING 快照，不把宿主清理造成的 cancelled 冒充任务的原始结果。

每次 resume 更新同一 run 的累计用量，不把中间 WAITING 与最终结果相加。`usage` 的主字段
只计主模型，`usage.compaction` 单列摘要调用；child 的用量仍属于 child run，当前列表不声称
覆盖整题成本。`state.output.usage` 保持空，Iris 用量只保存在上述结构中；它不会更新 Inspect
自身累计 token/cost 统计，因此 Inspect 的 token/cost 限额不约束 Iris。运行预算应通过
`AgentRunOptions` 指定，具体评分器独立读取自己需要的任务产物。

外部取消时先通知 Iris，再等待原 start/resume 调用退出；若准备阶段尚未创建 run，则直接
取消准备任务。取消后的等待和资源关闭会屏蔽 Inspect 的 AnyIO scope 重复取消；
子 Agent proxy 的 resume 取消沿用 harness 的结算方式。回调退出后，
接入层取消尚未处理的 WAITING 并调用 `runner.aclose()`；回调异常继续向 Inspect 传播。
关闭后不支持用导出的 run ID 恢复已释放的进程内会话。

## 验证

定向测试使用确定性 fake provider 和真实 SDK；默认执行及回调执行都经过 Inspect 的实际调度和日志写出，
评分关闭、模型使用占位配置。它们验证接口，不是 benchmark 成绩，也不调用真实模型。

```shell
uv run --group eval pytest tests/evals/test_iris_solver.py
uv run --group eval ruff check evals tests/evals
uv run --group eval mypy evals
```

未安装 `eval` 依赖组时，这组测试会跳过。

实现入口：[iris_solver.py](iris_solver.py)；SDK 说明：[Python SDK 入门](../docs/getting-started/python-sdk.md)和[运行时参考](../docs/reference/runtime.md)；
外部接口参考：[Inspect Solvers](https://inspect.aisi.org.uk/solvers.html)。
