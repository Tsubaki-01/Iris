# 离线汇总 Harbor 结果

本入口读取单个已存在的 Harbor job，输出逐 trial CSV、JSON 明细和 Markdown 报告。
它不调用模型或工具，不改写原始 reward。具体任务清单、实验参数和报告保存在 Git 忽略的 `evals/experiments/` 对应批次目录。

## 输入与命令

在仓库根目录准备 [Harbor 宿主环境](../harbor/README.md)，然后提供以下输入：

| 输入 | 含义 |
| --- | --- |
| `job_dir` | 一个 Harbor job 的输出目录 |
| `--manifest` | 本次任务清单 JSON 的路径 |
| `--attempts` | 每题计划执行次数，必须显式填写 |
| `--task-id` | 只汇总清单中的指定任务，可重复传入；省略时采用全部 `task_ids` |
| `--review-decisions` | 可选的显式评分复核 JSON 文件；省略时采用自动分类 |
| `--output` | 汇总材料输出目录 |

manifest 包含 `name`、`dataset_id`、`revision` 三个字符串字段，以及字符串数组 `task_ids`。
这些字段来自本次实际实验条件；脚本不内置题集或实验预算。

设置当前进程的 `EVAL_JOB_DIR`、`EVAL_MANIFEST`、`EVAL_ATTEMPTS` 和 `EVAL_REPORT_DIR` 后执行：

```powershell
uv run --project evals python -m evals.reporting "$env:EVAL_JOB_DIR" --manifest "$env:EVAL_MANIFEST" --attempts "$env:EVAL_ATTEMPTS" --output "$env:EVAL_REPORT_DIR"
```

每次输入一个 job。校准、正式运行、重跑或不同配置的 job 分开汇总。

## 应用显式复核

自动分类依据运行错误和已有 reward，无法确定每个 verifier 是否实际完成评分。
检查原始材料后，可按 `trial_name` 写入复核决定，再通过 `--review-decisions` 生成报告：

```json
{
  "trial-a": {
    "reviewed_score_status": "valid",
    "raw_reward": 0,
    "category": "output_budget_exhausted",
    "reason": "执行因输出预算结束，评分程序正常完成并给出零分。",
    "retry_as_infrastructure": false
  }
}
```

每条决定的五个字段均必填：

- `reviewed_score_status`：`valid` 表示原始二值评分有效；`invalid` 表示已有评分无效；`missing` 表示没有评分。
- `raw_reward`：与原始 trial 的 reward 一致，只有 `missing` 使用 `null`。它用于核对复核对象，不能覆盖原始分数。
- `category`、`reason`：非空的分类与复核依据，随报告保存。
- `retry_as_infrastructure`：布尔值，只记录重跑资格，不会触发任务执行。

```powershell
uv run --project evals python -m evals.reporting "$env:EVAL_JOB_DIR" --manifest "$env:EVAL_MANIFEST" --attempts "$env:EVAL_ATTEMPTS" --review-decisions "$env:EVAL_REVIEWS" --output "$env:EVAL_REPORT_DIR"
```

Python API `summarize_job()` 新增可选参数 `review_decisions_path: Path | None = None`。
复核字段无效或 `raw_reward` 与实际记录不符时，API 抛出 `ValueError`，CLI 报告参数错误并停止，不写输出。
一个复核文件可以供多个 job 共用；未匹配当前 job 的 trial 名列在 `unused_review_trials`，便于核对漏用或拼错的名称。
当前 job 的额外 trial 也可复核，但仍单列，不进入计划统计。

## 结果口径

输出为 `trials.csv`、`summary.json` 和 `report.md`，原始材料路径相对报告目录保存。

- 计划分母由任务清单和 attempts 确定；未启动任务保留空记录，额外 trial 单列。
- 当前成功统计采用 verifier 的二值 `reward`：有效 1 分为通过，有效 0 分为失败，缺失不补 0。
- `automatic_score_status` 保留自动分类；`reviewed_score_status` 保留显式复核状态，未复核时为空。
  `score_status` 是统计采用的状态：有复核时采用复核，否则采用自动分类。
- `reward` 和 `rewards` 始终是原始分数；复核的 `review_category`、`review_reason` 和 `retry_as_infrastructure` 独立保存。
- 模型执行、评分、清理和归档分别保留；预算耗尽仍采用 verifier 的实际结果。
- 费用和 token 未知时保持空值，已知合计同时给出覆盖数量。
- 同题多次执行按开始时间标记 occurrence，表示观察顺序，不代表受控随机 seed。

JSON 和 CSV 保存上述状态与复核字段；Markdown 另列显式复核表，展示自动判断、采用的复核状态及依据。
`invalid`、`missing` 和仍待复核的 `review_required` 均不进入有效评分数，但继续保留在计划分母中。

Iris 轨迹归档与可复查覆盖分别统计：后者还要求 trial 结果、verifier 输出日志，以及已有评分对应的原始 reward 文件。
任务产物采集独立展示。Harbor artifact manifest 的 `failed` 无法单独区分源不存在和下载失败，
汇总将这种情况标为 `unknown`，具体原因查 trial 日志；已记录下载成功但本地文件缺失时标为 `partial`。

## 验证

测试使用合成的结果文件，不需要题集、Docker 或模型：

```shell
uv run --project evals python -m pytest tests/evals/reporting/test_reporting.py
```

返回：[评测入口](../README.md) · [Harbor 接入](../harbor/README.md)。
