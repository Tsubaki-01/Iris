# 测试、示例与评测

验证应回答本次改动带来的问题。确定性测试适合检查契约与状态转换，真实模型示例适合确认服务集成，评测则需要明确题集与评价目标。它们的成功不能相互替代。

## 运行相关测试

在仓库根目录运行相关测试：

```shell
uv run pytest tests/harness/test_session_history.py
```

替换测试路径即可验证自己的改动。

## 从改动找到测试

| 改动 | 主要测试入口 |
| --- | --- |
| Agent YAML / 配置 | [tests/agents](../../tests/agents) |
| 上下文 / 压缩 | [tests/context](../../tests/context)、[tests/runtime](../../tests/runtime) 的 context/compaction 用例 |
| Runner / 恢复 / SessionManager / 历史 | [tests/harness](../../tests/harness) |
| 生命周期与存储 | [存储契约测试](../../tests/store/test_lifecycle_store_contract.py)、[tests/store](../../tests/store)及 harness 的状态转换测试 |
| 工具 / MCP / 命令 | [tests/tools](../../tests/tools)、[tests/mcp](../../tests/mcp)、[tests/command](../../tests/command) |
| 消息 / 模型 / 图片 | [tests/message](../../tests/message)、[tests/providers](../../tests/providers)、[图片处理](../../tests/utils/test_images.py) |
| 记忆 / 经验 / 目标 / 清单 | [tests/memory](../../tests/memory)、[tests/evolution](../../tests/evolution)、[tests/goal](../../tests/goal)、[tests/todo](../../tests/todo)及 harness 的相关集成测试 |
| 流式 / 观测 / 语音 | [tests/streaming](../../tests/streaming)、[tests/observability](../../tests/observability)、[tests/speech](../../tests/speech) |
| Inspect、Harbor 接入与评测汇总 | [评测入口](../../evals/README.md)、[tests/evals](../../tests/evals) |

先定位现有用例和复用的 provider/store 替身，再补一个能复现新需求的测试。不要为了一个消息映射错误搭起完整 UI、远端模型或多服务集成。

代码风格和类型配置在 [pyproject.toml](../../pyproject.toml)。例如修改一个 Python 文件后，可针对它运行 `uv run ruff check 路径` 与 `uv run mypy 路径`。`.pre-commit-config.yaml` 定义已有提交检查，不需要为本次文档建设新增一套门禁。

## 可运行示例的证据范围

| 方式 | 能证明什么 | 不能单独证明什么 |
| --- | --- | --- |
| 配置解析、导入或代码块编译 | 字段和语法可接受 | 真实功能完成 |
| 脚本化 provider + 真实 Runner/工具 | 接口接线、状态转换和本地副作用符合脚本 | 模型会自主选择正确行动 |
| 真实 provider 示例 | 某次环境中的协议、凭据和任务流程可运行 | 全部模型、长期稳定性或普遍任务成功率 |
| 明确题集的评测 | 所定义样本与指标上的表现 | 超出题集范围的能力 |

文档里的完整例子应交代凭据、文件、可选服务和工作目录。输出不确定时描述成功条件；不要用固定模型文本作为验收断言。

## Inspect AI 接入

[Inspect 接入](../../evals/inspect/README.md)提供仓库级 `iris_solver()`，入口为 `evals.inspect.solver`，让 Inspect 调用真实 AgentRunner、投影结果与用量并收尾资源。它不随 Iris Python 包发布，使用时在仓库根目录安装 `eval` 依赖组：

```shell
uv sync --group eval
```

当前接入不附带题集、benchmark adapter 或评分器。`eval` 组包含 Inspect AI 和用于取消收尾的 AnyIO；普通 Iris 用户无需安装。安装后可运行接入测试：

```shell
uv run --group eval python -m pytest tests/evals/inspect/test_solver.py
```

测试使用确定性 provider 和真实 SDK，验证默认输入、回调、状态与用量投影、取消、资源收尾及轨迹归档，不调用真实模型，也不代表 benchmark 成绩。调用 solver 时显式提供 `trajectory_dir`；每个 sample 保存独立的 `trace.json + assets/`，`.eval` 的 `iris_trace` 保留对应结构与索引。

增加评测时先明确想测的行为、固定输入与预算、结果记录和评分方式。报告实际运行条件，区分检索选对工具、工具执行正确与最终任务完成。

## Harbor 终端评测接入

[Harbor 接入说明](../../evals/harbor/README.md)提供独立宿主环境、Iris wheel 安装和容器 SDK 入口。宿主固定 Harbor 0.24.0，容器仅安装 Iris；两套依赖不混装。容器安装材料显式指定 Linux uv 可执行文件、Python patch 版本和依赖锁文件，下载最多额外重试两次。

SDK 入口测试使用仓库环境，adapter 契约与离线汇总测试使用 `evals/.venv` 环境，分别通过 `uv run --group eval` 和 `uv run --project evals` 执行。它们验证文件/命令执行与取消收尾、Harbor 安装参数与退出协议、已有结果汇总。测试专用 verifier 用于验证接线，正式题集和评分器另行选择。

[离线汇总](../../evals/reporting/README.md)读取已有 Harbor job，以外部传入的任务清单和运行次数计算结果。具体题集、实验参数和批次配置保存在 `evals/experiments/`，由 `evals/.gitignore` 排除；通用接入和确定性测试 fixture 保留在 `evals/` 与 `tests/evals/`。

需要人工确认评分有效性时，可传入显式复核决定。汇总保留原始 reward、自动判断和复核理由，以复核后的评分状态参与统计；复核不会修改原始结果或启动重跑。

真实容器验收需要 Linux Docker 引擎及镜像/依赖下载网络，并显式指定开关：

```shell
uv run --project evals python -m pytest tests/evals/harbor/test_integration.py --run-harbor-docker
```

它使用容器内确定性模型服务，经过真实 Harbor → Iris → 文件/命令 → verifier 链路，并在容器销毁后检查归档。默认运行测试时，这组容器场景跳过。

下一步：[贡献流程](index.md) · [文档维护](docs.md)。
