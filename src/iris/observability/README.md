# iris.observability

本包把 Iris 已有的执行事实转为标准 OpenTelemetry spans，通过 OTLP HTTP/protobuf 导出。
它不拥有 Run 状态、恢复、业务重试或存储；lifecycle/store 仍是权威来源。

SDK/CLI、子 Agent、Memory 和 Evolution 已接入普通与流式模型记录和正文投影。
前台 trace 包含每次 activation、请求准备、模型、真实工具、控制与 observer 区间；
后台维护使用独立的资源 cycle。

## 配置与使用

API 是直接依赖，导出 SDK 按需安装。在本仓库执行：

```powershell
uv sync --extra observability
```

Agent YAML 的采集策略默认关闭：

```yaml
observability:
  enabled: false
  capture_content: false
  max_content_chars: 65536
```

导出目标复用全局 `iris.init_config(observability=...)`。`traces_endpoint` 是完整 URL，
例如 `http://127.0.0.1:5000/v1/traces`；`headers` 默认空字典，`service_name` 默认 `iris`，
`timeout_seconds` 默认 5 秒。只解析配置不会加载 SDK 或连接服务。

宿主也可以借用已有的标准 OTel provider：

```python
from opentelemetry.trace import TracerProvider

from iris.observability import AgentObservabilityConfig, ObservabilityExportConfig
from iris.observability.service import Observability


def make_observability(host_tracer_provider: TracerProvider) -> Observability:
    return Observability.from_config(
        AgentObservabilityConfig(enabled=True, capture_content=True),
        ObservabilityExportConfig(),
        tracer_provider=host_tracer_provider,
    )
```

不传 `tracer_provider` 时，启用采集必须提供 endpoint 和可选依赖。自建服务由创建者
`await observability.aclose()`，在线程中排空和关闭 SDK；借用 provider 不 flush/shutdown。
本包不会替换进程全局 TracerProvider。

`AgentRunner.from_config*()` 和 `RuntimeFactory.from_config*()` 接受 `observability=`。
完整实例注入优先于 YAML，所有消费者借用；宿主在 runner、共享协调器与真实 worker
结束后最后关闭服务。未注入时，runtime 装配按 Agent 开关创建服务，environment 拥有其
关闭责任；直接 RuntimeFactory 调用者使用 `await runtime.environment.aclose()`。
Memory/Evolution 的构造器同样接受借用服务并包装自己的 raw provider，工厂只转发依赖。
完整服务注入 runner 时保留构建策略，不再重包。

`iris chat` 自动创建宿主共享实例，并在正常退出及构造、准备失败时清理。禁用采集不因
观测读取全局配置。Maintenance、Goal successor 和 deadline 只重置 OTel 上下文，
保留原有业务 ContextVars 和调度逻辑。

## 前台调用关系

一次 logical Run 可以经过多次等待与恢复；每次真实驱动建立独立 activation，使用已有
run/session IDs 关联。模型步骤通过 `iris.step.index` 关联 prepare、main 和工具，不建立
额外 step 节点。压缩模型是 prepare 的子节点，purpose 为 `compaction`。

工具节点包含实际处理和最终结果，middleware 恢复后按最终 `is_error` 判定；并行工具
各自完成时结束。未进入执行器的 preflight/人工决定只留下 `iris.tool.decision` 事件。
subagent 的等待正常结束本次区间；继续执行的 control 覆盖 child 与结果归一化，并在
父 Agent 下一 activation 前结束。恢复不会重放旧模型或旧工具节点。

`iris.driver.outcome` 描述实际驱动返回、失败或取消；`iris.run.status` 描述权威 Run 状态。
例如结果已提交 completed 后 observer 等待被取消，Run 仍为 completed。遥测不会修改
权威结果、触发业务重试或成为恢复依据。

## 配置来源与上下文诊断

[`facts.py`](facts.py) 定义 `SourceAdopted` 及其执行关联作用域，记录真实消费点采用的来源文本、
版本和身份。该路径独立于 OTel 开关，通过 runner 或维护协调器的 live publisher 发布；
关闭 trace 不等于关闭已配置的 live 事实。后台 worker 将事实送回绑定的 event loop，发布失败
只记日志，不影响原操作。

`ConfigurationApplied` 由 harness 提供，`ContextPreparation`、`ContextStage` 和
`ContextDecision` 由 runtime 提供。它们分别描述实际配置与请求准备，不成为恢复依据。
OTel 使用 `iris.configuration.snapshot_id`、`iris.context.preparation_id` 和流式调用的
`iris.model_stream.id` 关联对应事实；原始 typed facts 与 broker 的短字段投影不同，详见
[harness](../harness/README.md#live-publisher-组合) 和
[streaming](../streaming/README.md#观察范围与诊断事件)。

## 边界与记录语义

Memory 和 Evolution 各自持有独立维护根，只在取得资源锁后开始，并在真实 worker 排空、
锁释放后结束。模型用途区分 overview/flush/dream 和 experience/revision；
`iris.maintenance.result` 保留实际 stage/status，不将积压 bool 当成功标志。
empty、blocked、no_change、conflict 保持非错误；failed 标记维护失败。取消后已经提交的
结果保留原状态；模型成功后的发布失败不会回改模型节点。独立 SDK 调用没有维护 cycle
时仍记录模型，但不向任意宿主 span 添加维护事件。发布成功不等于其他 runner 已采用。

- 关闭时内部包装 helper 返回原 provider；scope 为空操作，不清空宿主上下文。
- wrapper 保留 complete-only / streaming 能力；估算不采集，也不发模型请求。
- stream 在首次消费时开始区间，每次建流、拉取和关闭短暂绑定上下文，yield 前恢复。
  正常 typed terminal、失败、取消、提前关闭分别记录；不会复制流或重拼响应全文。
- `capture_content` 和 span 采样都允许时才读取正文。消息保持 role/parts 顺序，system
  仍是输入消息；工具结果采用最终 `model_blocks`，包括错误文字、图片引用与 Hook 反馈。
- 超限内容省略标准 JSON 属性，改写 `iris.content.<完整属性键>.preview`，并在
  `iris.content.truncated_fields` 列出该键。图片只记录已有引用和 MIME，不读取或上传文件。
- 只在真实模型节点记录 provider 明确返回的用量。缺失/null 与真实 0 区分，流式快照取
  最新值；默认 0、`complete=True` 和 UI 派生总量都不能证明完整账单。
- 记录或导出故障进入标准库 logging，不重跑业务，不改变业务返回和取消。

`config.py` 只定义策略；`service.py` 统一标准 span/context、记录 gate 和 SDK 关闭；
`content.py` 负责 typed 内容投影；`provider.py` 拥有单次 complete/stream 的记录区间。
`facts.py` 仅提供采用事实和局部关联；上层 owner 决定业务范围、结果与发布出口，本包不建立
独立调度器、持久状态库或自定义 Span 类型。

内容格式采用 [GenAI schema 固定快照](https://github.com/open-telemetry/semantic-conventions-genai/tree/e07f4ebacb08f56db8c4c882d117720333fbca04)。

## 本地 MLflow 看板

MLflow 独立运行，不加入 Iris 依赖。本地验收使用 3.14.0；在仓库根目录的单独
PowerShell 终端启动：

```powershell
$env:NO_PROXY = '127.0.0.1,localhost'
$env:MLFLOW_MODEL_CATALOG_URI = ''
New-Item -ItemType Directory -Path 'tmp\observability-mlflow' -Force | Out-Null
uv tool run --from 'mlflow==3.14.0' mlflow server --backend-store-uri 'sqlite:///tmp/observability-mlflow/mlflow.db' --host 127.0.0.1 --port 5000
```

在 [本地 UI](http://127.0.0.1:5000) 创建 experiment 并取得 ID。第二个终端执行：

```powershell
$env:NO_PROXY = '127.0.0.1,localhost'
$observabilityExperimentId = Read-Host '输入 MLflow experiment ID'
uv run --extra observability python examples/observability/basic.py --experiment-id "$observabilityExperimentId"
uv run --extra observability python examples/observability/streaming_child.py --experiment-id "$observabilityExperimentId"
```

示例使用确定性模型，通过公开 SDK 执行真实工具和恢复流程。打开 experiment 的 Traces
查看实际父子树，在模型与工具节点查看 Inputs/Outputs、耗时和 Tokens；用 Sessions
查看同一会话的不同 activation。等待前后的 trace 不会自动合并。一个 trace 含父子多个
会话时，MLflow 的 trace 级 Session 可能归到 child；父 Session 列表不保证包含全部恢复
控制 trace。可在 Traces 打开调用树，并用 span 的 run/parent ID 核对关联。更多示例选项见
[示例说明](../../../examples/observability/README.md)。

`NO_PROXY` 仅让该终端的 localhost 请求直连，避免本机代理转发。MLflow 首次接收模型用量
可能同步读取远程模型目录；`MLFLOW_MODEL_CATALOG_URI=''` 使用其官方离线选项避免这项
网络等待。这些是验收进程的环境设置，不改用户持久环境，也不由 Iris 解析。

看板显示的 token 总量可能由已知输入/输出派生。判断“未知”与“真实零”时查看原始
`gen_ai.usage.*` 属性；总量不是完整账单。超限正文在自定义 preview 字段查看，标准 I/O
字段保持省略；本地图片仅保留 URI/MIME，浏览器不保证能展示像素。后端离线时业务继续，
导出器通过标准 logging 报告失败；正常关闭会排空 SDK，不需要逐 Run 强制 flush。

参考：[OTLP 接收](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/ingest/)、
[属性映射](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/attribute-mapping/)、
[模型目录开关](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.environment_variables.html#mlflow.environment_variables.MLFLOW_MODEL_CATALOG_URI)。

使用与设计：[观测接入](../../../docs/cookbook/observability.md) · [流式与观测参考](../../../docs/reference/streaming-observability.md)。
