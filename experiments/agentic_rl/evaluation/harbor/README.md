# Iris 的 Harbor 接入

本目录在任务容器内通过真实 Iris SDK 执行文本终端任务。容器入口不依赖 Harbor 或 Inspect；模型、工具循环、预算与 SQLite 状态由 Iris 管理。

## Harbor 宿主

从仓库根目录构建 Iris wheel，并准备锁定 Harbor 0.24.0 的独立宿主环境：

```shell
uv build --out-dir dist
uv sync --project experiments/agentic_rl
uv run --project experiments/agentic_rl harbor agent schema evaluation.harbor.agent:IrisHarborAgent
```

Harbor 宿主只依赖 Harbor，Iris wheel 安装到任务容器自己的 Python 3.12 环境。两者依赖链的 websockets 版本要求不一致，因此分开安装。

在所选任务的 `harbor run` 中，以 `-a evaluation.harbor.agent:IrisHarborAgent` 指定接入，`-m provider/model` 选择主模型，使用 `--ak` 传入下列参数：

| 参数 | 含义 |
| --- | --- |
| `iris_wheel` | 必填，本仓库 `uv build` 生成的 wheel 路径 |
| `profile_dir` | 自包含 profile 目录，默认使用本目录的 terminal profile |
| `max_model_steps` | 默认 20，Iris 的模型步数上限 |
| `cleanup_timeout_sec` | 默认 30，外层取消后确认进程退出和 SDK 收尾的等待时间 |
| `iris_revision` | 运行材料对应的源码 revision；省略时保留未知 |

通过 Harbor `--ae` 传入 Iris 所需的 provider 环境变量。执行过程继承 Harbor 的 agent user 和 scoped environment，adapter 仅补充模块路径；任务自己的 PATH 不改变。主模型名称由 `-m` 决定，协议与采样设置由 profile 决定。

正常或预算结束由 verifier 检查实际产物。普通运行失败走 Harbor 的非零 Agent 退出路径；资源收尾无法确认时走错误回收，跳过共享 verifier。结果和 token 用量在 Harbor 同步日志后回填，费用与 cache token 未知时保持空值。

宿主契约测试与 SDK 入口测试分别在各自依赖环境中运行：

```shell
uv run --project experiments/agentic_rl python -m pytest tests/experiments/harbor/test_agent.py
uv run --group eval python -m pytest tests/experiments/harbor/test_runner.py
```

## 容器入口

Linux 环境先安装本仓库构建的 Iris wheel，并将 `evaluation/` 放入 Python 可导入目录。使用独立解释器启动，不激活 venv，不改变任务自己的 PATH：

```shell
/installed-agent/iris/venv/bin/python -m evaluation.harbor.runner \
  --profile /installed-agent/iris/evaluation/harbor/profiles/terminal/agent.yaml \
  --model openai/example-model \
  --instruction-file /installed-agent/iris/instruction.txt \
  --workspace /task-project \
  --logs-dir /logs/agent/iris \
  --max-model-steps 20
```

凭据与端点使用 Iris 的 `IRIS_PROVIDER_API_KEYS__PROVIDER` 和
`IRIS_PROVIDERS__PROVIDER__BASE_URL`。`--model` 覆盖 profile 的主模型路由，其他采样和协议设置沿用 profile。

profile 在容器内加载，workspace 使用任务实际目录；命名模板写到 profile 旁的
`runtime-prompts/`，避免修改任务代码仓库。终端配置开放明确的文件和命令工具，默认关闭长期记忆、Goal、Todo 和自动经验维护。

## 停止与结果

SIGTERM 转为 SDK 停止请求：准备阶段取消原调用，已准入的运行请求取消，WAITING 按无人回答结束。关闭成功后才写入 `cleanup_completed=true`。

| 文件 | 内容 |
| --- | --- |
| `control.json` | 本 invocation 的 PID、session/run 身份 |
| `lifecycle.db` | 已提交的 Iris 运行事实 |
| `result.json` | 原始任务结果、清理状态、模型、用量和轨迹状态 |
| `trace.json`、`assets/` | 父子执行记录与可移动的引用产物 |

退出码 `0` 表示正常或预算结束，`1` 表示普通运行失败/取消/无人处理的 WAITING，`70` 表示资源收尾失败。仅轨迹 partial/failed 不改变原运行退出码。WAITING 的原始结果与收尾后取消的 store 记录分别保留。

本目录不定义正式题集或评分器。测试中的 verifier 仅验证接入链路；公共 benchmark 和付费模型运行由实验另行选择。
