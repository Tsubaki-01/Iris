# Iris 的 Harbor 接入

本目录在任务容器内通过真实 Iris SDK 执行文本终端任务。容器入口不依赖 Harbor 或 Inspect；模型、工具循环、预算与 SQLite 状态由 Iris 管理。

## Harbor 宿主

从仓库根目录构建 Iris wheel，并准备锁定 Harbor 0.24.0 的独立宿主环境：

```shell
uv build --out-dir dist
uv sync --project evals
uv run --project evals python -m harbor.cli.main agent schema evals.harbor.agent:IrisHarborAgent
```

`evals/pyproject.toml` 管理 Harbor 宿主依赖，环境位于 `evals/.venv`，不安装评测包。
从仓库根目录使用 `python -m harbor.cli.main` 启动 CLI，使其能导入 `evals.harbor`。
Iris wheel 安装到任务容器自己的 Python 3.12 环境。两者依赖链的 websockets 版本要求不一致，因此分开安装。

在所选任务的 `harbor run` 中，以 `-a evals.harbor.agent:IrisHarborAgent` 指定接入，`-m provider/model` 选择主模型，使用 `--ak` 传入下列参数：

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
uv run --project evals python -m pytest tests/evals/harbor/test_agent.py
uv run --group eval python -m pytest tests/evals/harbor/test_runner.py
```

## 真实容器验收

启用 Linux Docker 引擎，并允许下载测试镜像、Python 与 wheel 依赖后，从仓库根目录执行：

```shell
uv run --project evals python -m pytest \
  tests/evals/harbor/test_integration.py --run-harbor-docker
```

测试构建当前工作树的 Iris wheel，经 Harbor 的 Trial 生命周期完成安装、运行、验收和容器删除。模型请求由容器内确定性 HTTP 服务回答；实际文件工具、命令、SQLite、SIGTERM 和 verifier 都走真实实现。

fixture 使用 `/task-project` 工作目录和 Python 3.10，Iris 使用独立 Python 3.12。验收包括文件计算的正确/错误产物、两个并发 trial、超时后停止写入、清理失败跳过 verifier，以及容器销毁后读取归档图片和全文。另有 provider 错误和缺失 reward 场景，分别保留运行错误与验收缺失状态。正式题集和模型成绩不在这些测试中。

## 容器入口

Linux 环境先安装本仓库构建的 Iris wheel。adapter 将 `evals/__init__.py` 和 `evals/harbor/`
上传到 `/installed-agent/iris/evals/`，自包含 profile 上传到 `/installed-agent/iris/profile/`。
使用独立解释器启动，不激活 venv，不改变任务自己的 PATH：

```shell
PYTHONPATH=/installed-agent/iris /installed-agent/iris/venv/bin/python -m evals.harbor.runner \
  --profile /installed-agent/iris/profile/agent.yaml \
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

terminal profile 是通用终端示例，采样、上下文和压缩参数沿用 Iris 默认配置。
具体实验的参数写入 `evals/experiments/` 下的自包含 profile，通过 `--ak` 的 `profile_dir` 参数指定。
`trace.json` 和 `result.json` 的 `configuration` 保存实际采用值，便于离线比较。

## 停止与结果

SIGTERM 转为 SDK 停止请求：准备阶段取消原调用，已准入的运行请求取消，WAITING 按无人回答结束。关闭成功后才写入 `cleanup_completed=true`。

| 文件 | 内容 |
| --- | --- |
| `control.json` | 本 invocation 的 PID、session/run 身份 |
| `lifecycle.db` | 已提交的 Iris 运行事实 |
| `result.json` | 原始任务结果、清理状态、模型、用量和轨迹状态 |
| `trace.json`、`assets/` | 父子执行记录与可移动的引用产物 |

退出码 `0` 表示正常或预算结束，`1` 表示普通运行失败/取消/无人处理的 WAITING，`70` 表示资源收尾失败。仅轨迹 partial/failed 不改变原运行退出码。WAITING 的原始结果与收尾后取消的 store 记录分别保留。

adapter 确认收尾回执后才交回 Harbor；读取回执期间被取消也会完成同样的确认。原始结果与清理回执在归档前写入，若归档后的状态回写失败，保留先前回执和原退出码，并在 stderr 中记录写入错误。

接入层不定义评分器，测试中的 verifier 仅验证接入链路。
具体题集、实验 profile、批次配置和参数保存在 Git 忽略的 `evals/experiments/`，实际运行采用对应任务的 verifier。
已有结果通过[离线汇总入口](../reporting/README.md)读取。

返回：[评测入口](../README.md)。
