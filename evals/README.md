# Iris 评测

本目录保存通用的仓库级评测接入和离线汇总，不随 `iris` Python 包发布。
所有命令均从仓库根目录运行。

| 任务 | 入口 |
| --- | --- |
| 用 Inspect sample 调用 Iris SDK | [Inspect solver](inspect/README.md) |
| 在 Harbor 任务容器中运行 Iris | [Harbor 接入](harbor/README.md) |
| 汇总已有 Harbor job 的结果 | [离线汇总](reporting/README.md) |

## 目录职责

```text
evals/
├── inspect/       # Inspect solver 和接入说明
├── harbor/        # installed-agent、容器 SDK 入口和结果投影
│   └── profiles/  # 通用终端示例配置
├── reporting/     # 读取已有结果的离线汇总
├── experiments/   # 本地实验材料，由本目录 .gitignore 忽略
├── pyproject.toml # Harbor 宿主的独立依赖声明
└── uv.lock        # Harbor 宿主的依赖锁定
```

对应测试位于 `tests/evals/inspect/`、`tests/evals/harbor/` 和 `tests/evals/reporting/`。
具体任务清单、模型与预算参数、批次配置和实验说明保存在 `evals/experiments/`，由 `evals/.gitignore` 忽略。
原始数据、安装材料和历史运行记录继续保存在 `.private/`。
仓库只保留通用接入、接口说明、示例和确定性测试 fixture。

## 依赖环境

Inspect 与真实 Iris SDK 使用仓库主环境的 `eval` 依赖组；Harbor 调度端使用
`evals/pyproject.toml` 和 `evals/.venv`。两套依赖链的 `websockets` 版本要求不同，分别安装：

```shell
uv sync --group eval
uv sync --project evals
```

`evals/pyproject.toml` 只管理 Harbor 宿主依赖，不安装评测包。Harbor 命令通过 Python 模块入口启动，
让仓库根目录下的 `evals` 可被导入：

```shell
uv run --project evals python -m harbor.cli.main agent schema evals.harbor.agent:IrisHarborAgent
```

容器中的 Iris 单独安装本仓库构建的 wheel，不安装 Harbor 或 Inspect。安装使用显式提供的
Linux uv 可执行文件、Python patch 版本和依赖锁文件，具体准备方式见 [Harbor 接入](harbor/README.md)。
Inspect solver 入口为 `evals.inspect.solver`；Harbor 接入为 `evals.harbor.agent:IrisHarborAgent`；
离线汇总入口为 `python -m evals.reporting`。导入 `evals` 本身不会加载这些可选依赖。

## 选择验证范围

Inspect 和容器 SDK 入口在主环境验证，Harbor adapter 与离线汇总在独立宿主环境验证：

```shell
uv run --group eval python -m pytest tests/evals/inspect/test_solver.py tests/evals/harbor/test_runner.py
uv run --project evals python -m pytest tests/evals/harbor/test_agent.py tests/evals/reporting/test_reporting.py
```

真实 Docker 接入测试需显式启用，具体命令见 [Harbor 验证](harbor/README.md#真实容器验收)。
这些接入测试使用确定性 provider；正式评测成绩由对应 benchmark 的 verifier 产生。
