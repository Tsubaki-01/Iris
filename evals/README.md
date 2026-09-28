# Iris 的评测实验与 Inspect AI 接入

本目录提供仓库级 `iris_solver()`：让 Inspect 调用真实 `AgentRunner`，取得输出、
运行状态和已有用量。该 Inspect 接入目前没有题集、benchmark 适配器或评分器。
另有独立的 Jev 工具检索实验，不依赖 Inspect，也不接入默认 Agent 执行流程。
本目录不随 `iris` Python 包发布，使用时从仓库根目录执行。

## Jev 工具检索最小实验（Noul）

[`jev_tool_search.py`](jev_tool_search.py) 复用当前 `DeferredToolIndex` 和八个内置工具的
真实定义：五个文件工具、`web_search`、`web_fetch`、`ask_question`。工具只用于构造目录，
不会执行。题集是预先编写的 [30 条中英文用例](fixtures/jev_tool_search.json)，包含
24 个正例和 6 个无需工具的负例；不是用户会话或业务 benchmark。

在根目录 `.env` 设置 `IRIS_PROVIDER_API_KEYS__TYPESAFE`，继续由 `iris.config.init_config`
读取，不需要新增依赖。以下命令会实际调用 TypeSafe API 并产生费用：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run python -X utf8 -m evals.jev_tool_search --pool bm25
uv run python -X utf8 -m evals.jev_tool_search --pool all
```

- `bm25`：从同一目录本地召回最多 5 个工具，Jev 只重排这些候选；空候选不发请求。
- `all`：Jev 直接判断全部 8 个工具，用于区分词法召回损失和模型判断错误。
- 固定使用 `jev-1.13.0`，每个候选一个 Noul 问题，在一次请求中提交。问题只包含查询和
  工具名称、完整描述、分组、标签，人工参考答案只留在本地计分。
- 按匹配概率降序排列，固定 `0.5` 阈值，最多返回 3 个工具。同分保持原候选顺序；
  阈值和问题没有根据这次结果调参。HTTP 或响应错误直接结束，不回退成成功结果。
- 默认输出 `tmp/jev-tool-search-bm25.json` / `tmp/jev-tool-search-all.json`，可用 `--output`
  指定路径。逐题保存候选、概率、实际模型、API token 用量和耗时；完整完成后写入汇总及
  按语言统计。失败中断的文件只有已完成题目，不代表完整实验。

正例分别统计候选召回率和 Top-1 / Top-3 命中率；负例统计返回非空列表的比例。Jev 指标包含
阈值过滤，本地基线只做检索。负例只是额外的分类探针，**不代表现有 Agent 在闲聊时会调用工具**。
耗时为本机实测，Jev 包含 HTTP 请求及解析，连接在整轮实验复用，首次连接耗时也纳入统计。
费用按官方每百万输入 token 0.042 美元估算，输出免费，不是账单核对结果。

2026-09-28 首次实测（单次运行，八工具小目录）：

| 指标 | 本地搜索 | BM25 候选 + Jev | 全目录 + Jev |
| --- | --- | --- | --- |
| 正例 Top-1 | 17/24 | 21/24 | 24/24 |
| 正例 Top-3 | 19/24 | 21/24 | 24/24 |
| 负例返回非空 | 5/6 | 0/6 | 0/6 |
| 单次请求耗时中位数 | 约 0.14 ms | 315 ms | 318 ms |
| 实际 API 请求数 | 0 | 29 | 30 |
| 估算输入费用 | 0 | $0.001094 | $0.001878 |

BM25 + Jev 剩下三个正例都在初筛漏掉了正确工具。这说明应同时评估召回与重排；小目录的
全量判断值得继续试验，但这里没有验证大目录、多轮对话、最终任务完成率或长期稳定性。
排第一正确也不意味着返回的其他候选都正确。

离线验证不读取 `.env`，不产生 API 费用：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run pytest tests/evals/test_jev_tool_search.py -p no:cacheprovider --basetemp="$PWD\tmp\pytest-jev"
uv run ruff check evals/jev_tool_search.py tests/evals/test_jev_tool_search.py
uv run mypy evals/jev_tool_search.py
```

接口依据：[TypeSafe API](https://docs.typesafe.ai/api)、[Noul](https://docs.typesafe.ai/primitives/noul)、
[模型与价格](https://docs.typesafe.ai/models)。

## 规则、Jev Noul / Choice 与 DeepSeek 对照

[`tool_search_comparison.py`](tool_search_comparison.py) 复用同一题集、工具目录和 Jev 请求，
加入当前 [`examples/chat/agent.yaml`](../examples/chat/agent.yaml) 配置的 DeepSeek。
凭据继续通过 Iris 配置和 provider factory 解析；实验直接用 HTTP 调用两个服务，保留
DeepSeek 原始的缓存命中与未命中 token 字段，不更改正式 provider 的响应契约。
**所有成本对照统一假设输入完全未命中缓存**，实际缓存字段只用于记录测量环境。

### 运行与比较范围

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run python -X utf8 -m evals.tool_search_comparison
```

默认每条查询重复三轮，并同时测试 BM25 最多五候选和全部八工具。题目顺序、两种候选范围
及 Noul、Choice、DeepSeek 的先后顺序由固定种子 `20260928` 打乱，串行交错调用，复用 HTTP
连接。包括规则在内共四组方法，每个方法、每个候选范围各有 90 次观测，但仍只有
**30 个独立查询**，完整默认运行产生 720 个观测。
规则基线是现有 BM25-like 算法，不包含针对题集追加的关键词表。

### 请求与回答格式

每条查询都是独立请求，输入为当前 query、候选工具元数据和固定选择指令，**没有历史消息**。
工具元数据只有 `name`、完整 `description`、`group`、`tags`，不包含参数 schema；历史用户
消息、模型回答、工具执行结果和记忆均不发送。三轮重复也不会累积会话。题目中的“刚才看过的
README”只是文字前提，没有对应的实际读取记录。

| 方法 | 请求组织 | 返回的选择 |
| --- | --- | --- |
| DeepSeek | 两条 `messages`：`system` 放固定指令，`user.content` 是包含 `query`、`tools` 的 JSON 字符串 | 最多三个工具名，按相关性排列 |
| Jev Noul | `state` 只有 `query`，`questions` 中每个工具各一道判断题 | 按逐项概率筛选并排序 |
| Jev Choice | `state` 只有 `query`，`questions.selection.criteria` 包含全部候选及 `__none__` | 一个工具或空列表 |

DeepSeek 沿用 YAML 中的模型名、temperature、top_p、max_tokens 和 timeout，请求显式设置
`response_format: {"type": "json_object"}`。具体的 `{"tools": ["tool_name"]}` 结构由 system
prompt 要求，返回后由 Pydantic 校验，并检查工具名属于候选集合；没有向 API 提交
`json_schema`。回答位于普通 `assistant.content`，没有调用参数、理由或置信度。

本轮保存的真实回答示例：

| query | `assistant.content` |
| --- | --- |
| 编辑已读取的配置文件，把 debug=true 替换为 debug=false | `{"tools": ["edit_file"]}` |
| 刚才看过的 README 中把旧项目名称改成 Iris，其他内容保留 | `{"tools": ["edit_file", "read_file"]}` |
| Thanks for editing the file earlier. No more changes are needed. | `{"tools": []}` |

Jev 分别测试：

- `jev_noul`：每个工具一道匹配判断题，固定 0.5 阈值，按概率返回最多三个工具。
- `jev_choice`：全部候选放入同一道 Choice，额外加入 `__none__` 选项；只返回首选工具，
  或在首选为 `__none__` 时返回空列表。不使用概率或置信度阈值，保存完整分布与置信度。

几组收到相同的查询和候选元数据；人工答案只用于本地计分。主要比较 Top-1 与负例误选。
Choice 本轮没有定义多选策略，因此 `top3` 记为 `null`，不将概率前三名解释为三个适用工具。
当前题集每条正例只标注一个工具，尚不能评价真正的多工具选择效果。**Top-1 命中只说明
第一个工具正确**，不会惩罚后面多返回的工具；上例额外返回 `read_file` 仍计作首选命中。

结果保存到 `tmp/tool-search-choice-comparison.json`，包含实际返回模型、逐次耗时、原始响应、
token 用量、首选变化次数和按候选范围汇总的质量、P50、P95、费用。支持 `--agent-config`、
`--env-file`、`--repeats` 和 `--output`。空候选不调用模型；API/解析失败会停止并保留已完成
记录，不伪装为拒答或规则结果。只有带 `completed_at` 和 `summary` 的文件是完成实验。

### 无缓存成本口径

费用按 2026-09-28 官方价格快照计算，DeepSeek 价格对应**实际返回的 `deepseek-flash`**：

| 每百万 token（美元） | Jev | DeepSeek 闲时 | DeepSeek 高峰 |
| --- | --- | --- | --- |
| 全部输入，按未命中缓存计价 | 0.042 | 0.15 | 0.30 |
| 输出 | 0 | 0.60 | 1.20 |

DeepSeek 闲时费用为 `(input_tokens × 0.15 + output_tokens × 0.60) / 1,000,000`，高峰翻倍；
Jev 为 `input_tokens × 0.042 / 1,000,000`。按选择次数归一化到每千次，包含空候选导致的
零请求观测。报告以 `cost_assumption: "all_input_tokens_cache_miss"` 标明口径，汇总费用
字段为 `no_cache_usd_off_peak`、`no_cache_usd_peak` 及对应的 `no_cache_usd_per_1000_*`。
`cache_hit_tokens` 和 `cache_miss_tokens` 保留供应商实际值，但不参与费用计算。

这只对已测 token 用量重新计价，**没有重新测量无缓存延迟**，也不是供应商实际扣费。
P50/P95 仍来自原始请求的实际缓存环境；高峰费用同样是价格情景，没有高峰延迟实测。

当前 YAML 请求 `deepseek-chat`，首个真实请求的返回模型为 `deepseek-flash`；两者分别记录，
不将其描述为历史版本的 DeepSeek Chat。选择阶段的性能也不代表完整 Agent 的性能：没有
执行工具、生成最终答案或衡量主模型是否因工具推荐而少走了一轮。

### 同期实测结果

加入 Choice 后重新交错运行全部四组，2026-09-28 北京时间 20:19:59—20:24:14 的全目录
结果如下，每组正例 72 次、负例 18 次。表中不混入此前仅有 Noul 的实验结果。

| 方案 | 正例首选命中 | 负例误选 | 平均输入 token | 实测 P50 / P95 | 无缓存每千次费用，美元 |
| --- | --- | --- | --- | --- | --- |
| 纯规则 | 51/72 | 15/18 | 0 | 0.17 / 0.25 ms | 0 API 费用 |
| Jev Noul | 72/72 | 0/18 | 1,491 | 359 / 506 ms | 0.0626 |
| Jev Choice | 72/72 | 0/18 | 929 | 355 / 451 ms | 0.0390 |
| DeepSeek | 72/72 | 0/18 | 423 | 563 / 804 ms | 闲时 0.0682；高峰 0.1365 |

Choice 的输入量和费用比逐项 Noul 降低 **37.7%**，首选质量在这份题集上相同。
Choice 中位延迟比 DeepSeek 低约 37%；在统一无缓存费用口径下，比 DeepSeek 闲时费用
低约 **42.8%**。这些结果只覆盖单轮选择，没有证明完整 Agent 的总节省。

BM25 五候选范围内，三种语义方法均为 63/72，仍受初筛漏召回限制。Choice 每千次费用
0.0268 美元，低于本轮 Noul 的 0.0365 和 DeepSeek 无缓存闲时的 0.0432，较后者低约 37.9%。
两种候选范围合计 531 次实际 API 调用，按无缓存闲时口径估算总费用 **$0.024871**。
三轮重复不增加独立样本数，也没有验证
多工具选择、置信度阈值策略或完整 Agent 的任务成本。

实际缓存记录：全目录三轮中，每轮 30 次请求均报告命中 256 个输入 token，合计命中率
60.5%；第一轮也不是冷缓存基线。BM25 每轮 29 次请求中，18 次命中 128 token、11 次为零。
这些字段来自 API usage，脚本每次都请求模型，并未复用整条回答。DeepSeek 的输入顺序是
固定 system 指令、query、工具目录，不能仅凭后面的目录相同就认定那些 token 的缓存来源；
具体命中了哪些文本，当前响应没有提供。此命中率不作为真实业务流量的假设或成本折扣。

### 离线检查

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run pytest tests/evals/test_jev_tool_search.py tests/evals/test_tool_search_comparison.py -p no:cacheprovider --basetemp="$PWD\tmp\pytest-jev-compare"
uv run ruff check evals/jev_tool_search.py evals/tool_search_comparison.py evals/_tool_search_metrics.py evals/_typesafe.py tests/evals/test_jev_tool_search.py tests/evals/test_tool_search_comparison.py
uv run mypy evals/jev_tool_search.py evals/tool_search_comparison.py evals/_tool_search_metrics.py evals/_typesafe.py
```

价格与接口依据：[DeepSeek 定价](https://api-docs.deepseek.com/quick_start/pricing/)、
[DeepSeek JSON 输出](https://api-docs.deepseek.com/guides/json_mode/)、
[TypeSafe 模型与价格](https://docs.typesafe.ai/models)、
[TypeSafe Choice](https://docs.typesafe.ai/primitives/choice)。

## 安装与配置

在仓库根目录安装独立的评测依赖组：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
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

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run --group eval pytest tests/evals -p no:cacheprovider --basetemp="$PWD\tmp\pytest-evals"
uv run --group eval ruff check evals tests/evals
uv run --group eval mypy evals
```

未安装 `eval` 依赖组时，这组测试会跳过。Windows 下 pytest 临时目录若出现 ACL 错误，
按项目测试规范在允许提权的环境中运行。

实现入口：[iris_solver.py](iris_solver.py)；SDK 说明：[harness](../src/iris/harness/README.md)；
外部接口参考：[Inspect Solvers](https://inspect.aisi.org.uk/solvers.html)。
