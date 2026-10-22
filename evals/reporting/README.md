# 离线汇总 Harbor 结果

本入口读取单个已存在的 Harbor job，输出逐 trial CSV、JSON 明细和 Markdown 报告。
它不调用模型或工具，不改写原始 reward。具体任务清单和实验参数保存在 Git 忽略的 `evals/experiments/`，报告保存在 `.private/`。

## 输入与命令

在仓库根目录准备 [Harbor 宿主环境](../harbor/README.md)，然后提供以下输入：

| 输入 | 含义 |
| --- | --- |
| `job_dir` | 一个 Harbor job 的输出目录 |
| `--manifest` | 本次任务清单 JSON 的路径 |
| `--attempts` | 每题计划执行次数，必须显式填写 |
| `--task-id` | 只汇总清单中的指定任务，可重复传入；省略时采用全部 `task_ids` |
| `--output` | 汇总材料输出目录 |

manifest 包含 `name`、`dataset_id`、`revision` 三个字符串字段，以及字符串数组 `task_ids`。
这些字段来自本次实际实验条件；脚本不内置题集或实验预算。

设置当前进程的 `EVAL_JOB_DIR`、`EVAL_MANIFEST`、`EVAL_ATTEMPTS` 和 `EVAL_REPORT_DIR` 后执行：

```powershell
uv run --project evals python -m evals.reporting "$env:EVAL_JOB_DIR" --manifest "$env:EVAL_MANIFEST" --attempts "$env:EVAL_ATTEMPTS" --output "$env:EVAL_REPORT_DIR"
```

每次输入一个 job。校准、正式运行、重跑或不同配置的 job 分开汇总。

## 结果口径

输出为 `trials.csv`、`summary.json` 和 `report.md`，原始材料路径相对报告目录保存。

- 计划分母由任务清单和 attempts 确定；未启动任务保留空记录，额外 trial 单列。
- 当前成功统计采用 verifier 的二值 `reward`：有效 1 分为通过，有效 0 分为失败，缺失不补 0。
- 模型执行、评分、清理和归档分别保留；预算耗尽仍采用 verifier 的实际结果。
- 费用和 token 未知时保持空值，已知合计同时给出覆盖数量。
- 同题多次执行按开始时间标记 occurrence，表示观察顺序，不代表受控随机 seed。

Iris 轨迹归档与可复查覆盖分别统计：后者还要求 trial 结果、verifier 输出日志，以及已有评分对应的原始 reward 文件。
任务产物采集独立展示。Harbor artifact manifest 的 `failed` 无法单独区分源不存在和下载失败，
汇总将这种情况标为 `unknown`，具体原因查 trial 日志；已记录下载成功但本地文件缺失时标为 `partial`。

## 验证

测试使用合成的结果文件，不需要题集、Docker 或模型：

```shell
uv run --project evals python -m pytest tests/evals/reporting/test_reporting.py
```

返回：[评测入口](../README.md) · [Harbor 接入](../harbor/README.md)。
