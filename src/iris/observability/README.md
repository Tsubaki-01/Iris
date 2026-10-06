# iris.observability

本包把 Iris 已有的执行事实转为标准 OpenTelemetry spans，通过 OTLP HTTP/protobuf 导出。
它不拥有 Run 状态、恢复、业务重试或存储；lifecycle/store 仍是权威来源。

SDK/CLI、子 Agent、Memory 和 Evolution 已接入普通与流式模型记录和正文投影。
当前尚未增加 activation、工具与维护 cycle 区间。

## 配置与使用

API 是直接依赖，导出 SDK 按需安装：

```powershell
uv add "iris[observability]"
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

## 边界与记录语义

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
上层 owner 决定业务范围和结果，本包不定义第二套事件总线、Span 类型或状态模型。

内容格式采用 [GenAI schema 固定快照](https://github.com/open-telemetry/semantic-conventions-genai/tree/e07f4ebacb08f56db8c4c882d117720333fbca04)。
