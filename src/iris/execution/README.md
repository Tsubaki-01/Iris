# 命令执行

本包定义 Native 与可选本地 Docker 命令环境的配置和服务契约。当前可以直接调用 Native 后端；Docker 后端和 Agent 命令注册将在后续阶段接入。

```python
from iris.execution import ExecutionConfig

native = ExecutionConfig()  # 默认 120 秒
docker = ExecutionConfig.model_validate({"mode": "docker"})
```

`DockerConfig` 默认使用预先准备的 `python:3.12-slim` 镜像、`none` 网络、2 CPU、1024 MiB 内存和 128 PID。`endpoint` 只接受本地 Unix socket 或 Windows named pipe。配置解析和导入本包都不连接 Docker；Native 模式显式声明 `docker` 块会报错。

`CommandService.execute(scope, request)` 接收已解析的宿主 cwd 和最终命令期限，返回前台执行事实。服务不拥有工具权限、生命周期历史或模型调用。

停止分为两个完成点：同步 `stop(scope)` 立即登记并调度操作；工具 body 只等待 `wait_stopped()` 的物理停止证明，外层结算等待 `wait_drained()` 确认旧调用收尾。`ExecutionStopReceipt` 仅标识一次已证实的停止，不包含 Future 或资源句柄，不持久化。消费旧收据不得再次停止之后重启的环境。

`CommandStopSlot` 是当前调用因果链的可写共享状态。context 投影保留槽的 identity，避免工具结果被 middleware 替换后丢失清理证明；它不是模型输入或历史数据。

## 直接调用 Native

```python
import asyncio
from pathlib import Path

from iris.execution import CommandRequest, ExecutionScope
from iris.execution.native import NativeCommandService


async def main() -> None:
    workspace = Path.cwd().resolve()
    service = NativeCommandService(workspace)
    try:
        await service.prepare()
        result = await service.execute(
            ExecutionScope(run_id="example", session_id="local"),
            CommandRequest(
                call_id="hello", command="echo hello", cwd=workspace, timeout_seconds=5
            ),
        )
        print(result.status, result.exit_code, result.stdout)
    finally:
        await service.aclose()


asyncio.run(main())
```

Windows 使用系统目录中的 `cmd.exe`，POSIX 使用 `/bin/sh`。每次命令创建新 shell，继承宿主环境和 PATH，无交互 stdin、无 TTY，不保留上次的 `cd` 或环境变量修改。Windows 进程隐藏窗口，host 必须使用支持 subprocess 的事件循环；库不更改全局 event loop policy。

Native **不是 OS 级隔离沙箱**。直接服务调用接收已解析的宿主目录，不代替工具层的 workspace 和权限裁决。普通退出保留实际退出码，包括 124/137；期限终止使用独立的 `timed_out` 状态。stdout/stderr 合计保留 1 MiB，继续排空超额数据，以 UTF-8 replacement 解码。前台退出后管道最多再排空 1 秒，后台继承管道时标记可能截断并返回。

单命令期限仅停止当前调用。`stop(scope)` 停止该 session 的当前调用，不影响独立 session；`aclose()` 停止所有仍持有的调用并拒绝新执行。POSIX 对本命令组发 TERM、有限等待后 KILL；Windows 用隐藏的 taskkill 尽力终止子树，再确认所持有前台退出。服务不追踪历次命令留下的后台程序，也不承诺回收宿主所有后代。重复取消不会打断已经开始的必要收尾；停止无法确认时抛出公共 unknown 并保留可重试的清理状态。

当前实际验证为 Windows / Python 3.12：中文与空格 cwd、stdout/stderr、真实退出码、4 MiB 输出、局部超时、重复取消、启动中的取消、后台继承管道、session 停止范围、旧收据不会停止新调用，以及关闭后拒绝新命令。Linux 进程组实测尚未完成，不能用控制测试替代该平台证据。
