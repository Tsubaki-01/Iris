# 观察模型、工具与运行过程

运行状态回答“任务现在怎样”；trace 回答“时间花在哪里、哪个调用产生了什么结果”。Iris 通过 OpenTelemetry 记录后者，运行恢复仍使用 lifecycle store。

## 启用 OTLP 导出

前提是已有接收 OTLP HTTP/protobuf traces 的服务。可选导出依赖与 Agent 配置分别设置。在仓库根目录执行：

```powershell
uv sync --extra observability
$env:IRIS_OBSERVABILITY__TRACES_ENDPOINT = "http://127.0.0.1:4318/v1/traces"
$env:IRIS_OBSERVABILITY__SERVICE_NAME = "iris-local"
```

上述地址是本地接收器的配置示例；你需要先启动对应服务，Iris 不会替你部署它。在已有 `agent.yaml` 中增加：

```yaml
observability:
  enabled: true
  capture_content: false
```

启动 `uv run iris chat agent.yaml`，提交一次会使用工具的任务。在接收器里查找 `service.name=iris-local`，应能看到 activation、模型和工具等调用区间；实际有没有导出成功，还要检查接收器与应用日志。

`enabled` 没有导出 endpoint 或注入的 tracer provider 时，会在装配时明确报错。只安装依赖不会自动启用采集，只启用采集也不会生成可视化看板。

## 按需要采集正文

要排查具体请求与响应，可以增加：

```yaml
observability:
  enabled: true
  capture_content: true
  max_content_chars: 65536
```

正文采集默认关闭。启用后按长度上限记录可支持的内容；观测中的图片为引用信息，不是完整像素备份。已知 token 会被记录，服务没有返回的用量不应解释成已测得的零值。

普通模型调用、摘要和后台维护各自有用途与归属，不能只把整个 trace 内所有数字当作单次用户回答的主模型费用。

## 宿主共享自己的 tracer provider

已有 OTel 基础设施的应用可以构造 `Observability.from_config(capture_config, export_config, tracer_provider=...)`，再通过 Runner 的 `observability=` 注入。该服务从 `iris.observability.service` 导入；配置模型从 `iris.observability` 导入。

注入的服务由创建者管理，Runner 不关闭借用的 provider。若 Observability 自行创建了 SDK，创建者在所有使用者退出后 `await observability.aclose()`；若 tracer provider 本身由宿主创建，则由宿主关闭它。Iris 不替换进程的全局 OTel provider。

## 先用确定性示例熟悉调用树

仓库 [observability 示例](../../examples/observability/README.md)使用脚本化 provider 驱动真实 Runner 和工具，不消耗模型 API key。`basic` 会读取一份真实文件，再产生回答。

若你使用示例配套的 MLflow 接收器，先创建 experiment 并取得 ID，然后执行（把 `EXPERIMENT_ID` 替换为实际值）：

```powershell
uv run --extra observability python -m examples.observability.basic --experiment-id EXPERIMENT_ID
```

脚本默认 endpoint 为 `http://127.0.0.1:5000/v1/traces`，可用 `--endpoint` 修改；它也不会创建接收器。`--disabled` 可运行同一业务流程但关闭观测，`--no-content` 仅采集元数据。

这类示例证明接线和记录方式，不是实际模型质量或远端导出成功的替代证据。

## 从正常故障入手排查

| 现象 | 应查看的材料 |
| --- | --- |
| 运行失败 | `RunResult.error` 与停止原因，再定位对应模型或工具 span |
| 长对话突然变慢 | 是否发生压缩，压缩耗时与主模型调用是否分开 |
| 工具已经显示文字但任务没有完成 | live partial 与 durable 终态，不能只看界面输出 |
| 看板无记录 | 采集开关、导出 endpoint、可选依赖、接收器状态和进程收尾日志 |
| 后台记忆整理产生额外调用 | maintenance 的独立周期与模型用途，而不是当前前台 Run 的调用树 |

精确配置与服务接口见[参考](../reference/streaming-observability.md)，职责关系见[设计解释](../design/streaming-observability.md)。
