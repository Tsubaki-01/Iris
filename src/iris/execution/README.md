# 命令执行

本包定义 Native 与可选本地 Docker 命令环境的配置和服务契约。当前已交付配置与内部类型，后端和 Agent 命令注册将在后续阶段接入。

```python
from iris.execution import ExecutionConfig

native = ExecutionConfig()  # 默认 120 秒
docker = ExecutionConfig.model_validate({"mode": "docker"})
```

`DockerConfig` 默认使用预先准备的 `python:3.12-slim` 镜像、`none` 网络、2 CPU、1024 MiB 内存和 128 PID。`endpoint` 只接受本地 Unix socket 或 Windows named pipe。配置解析和导入本包都不连接 Docker；Native 模式显式声明 `docker` 块会报错。

`CommandService.execute(scope, request)` 接收已解析的宿主 cwd 和最终命令期限，返回前台执行事实。服务不拥有工具权限、生命周期历史或模型调用。

停止分为两个完成点：同步 `stop(scope)` 立即登记并调度操作；工具 body 只等待 `wait_stopped()` 的物理停止证明，外层结算等待 `wait_drained()` 确认旧调用收尾。`ExecutionStopReceipt` 仅标识一次已证实的停止，不包含 Future 或资源句柄，不持久化。消费旧收据不得再次停止之后重启的环境。

`CommandStopSlot` 是当前调用因果链的可写共享状态。context 投影保留槽的 identity，避免工具结果被 middleware 替换后丢失清理证明；它不是模型输入或历史数据。
