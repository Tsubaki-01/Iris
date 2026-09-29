# 命令执行

本包提供 Native 与可选本地 Docker 命令环境。两种后端可直接通过服务调用，也可由 Agent 显式注册 `exec_command` 使用；普通工具仍在宿主执行。

```python
from iris.execution import ExecutionConfig

native = ExecutionConfig()  # 默认 120 秒
docker = ExecutionConfig.model_validate({"mode": "docker"})
```

`DockerConfig` 默认使用预先准备的 `python:3.12-slim` 镜像、`none` 网络、2 CPU、1024 MiB 内存和 128 PID。`endpoint` 只接受本地 Unix socket 或 Windows named pipe。配置解析和导入本包都不连接 Docker；Native 模式显式声明 `docker` 块会报错。

`CommandService.execute(scope, request)` 接收已解析的宿主 cwd 和最终命令期限，返回前台执行事实。服务不拥有工具权限、生命周期历史或模型调用。

停止分为两个完成点：同步 `stop(scope)` 立即登记并调度操作；工具 body 只等待 `wait_stopped()` 的物理停止证明，外层结算等待 `wait_drained()` 确认旧调用收尾。`ExecutionStopReceipt` 仅标识一次已证实的停止，不包含 Future 或资源句柄，不持久化。消费旧收据不得再次停止之后重启的环境。

`CommandStopSlot` 是当前调用因果链的可写共享状态，保留 receipt 与尚未完成的 cleanup_error。context 投影保留槽的 identity，避免工具结果被 middleware 替换后丢失清理证明；它不是模型输入或历史数据。

`ExecutionBinding` 绑定配置、共享服务与 `CommandEnvironment`。root 装配一次，child 借用同一绑定和调用槽，不单独关闭服务。Agent 配置与权限示例见 [agents](../agents/README.md)，工具参数与结果见 [tools](../tools/README.md)。

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

当前已用 Windows / Python 3.12.12 与 WSL Ubuntu / Python 3.12.3 的真实进程验证：中文与空格 cwd、stdout/stderr、真实退出码、4 MiB 输出、局部超时、重复取消、启动中的取消、后台继承管道、session 停止范围、旧收据不会停止新调用，以及关闭后拒绝新命令。Linux 还验证了前台响应 TERM 后，同组忽略 TERM 的子进程仍被 KILL；这不扩展为历次后台进程的回收保证。

## 本地 Docker

开发环境先准备 Linux Docker Engine，或 Windows Docker Desktop 的 Linux engine，再显式安装 extra 和准备镜像：

```console
uv sync --extra sandbox
docker pull python:3.12-slim
```

服务不会安装 Docker、拉取镜像或回退 Native。基础导入不加载 aiodocker；仅 Docker 的 `prepare()` 导入驱动并检查本地引擎、Linux 容器模式和预备镜像。默认 endpoint 在 Windows 为 `npipe:////./pipe/docker_engine`，Linux 为 `unix:///var/run/docker.sock`；显式传给驱动，不由远程 context 或 `DOCKER_HOST` 改写。

在前面的调用示例中，可将服务构造替换为：

```python
from iris.execution import DockerConfig
from iris.execution.docker import DockerCommandService

service = DockerCommandService(workspace, DockerConfig(), workspace_writable=True)
```

一个服务只拥有一个容器，同服务的 session 与 child scope 共用文件、依赖、后台服务和资源额度；两个服务各自独立。首次实际命令才创建和启动容器，正常退出不会停容器。root 目录单一挂载到 `/workspace`，child cwd 相对 root 投影；命令可以访问整个 root 挂载，不获得每个 session/child 独立的目录隔离。挂载只读由 `workspace_writable` 决定，授权模型命令不等于逐次拦截 shell 内部文件操作。

镜像需提供 Python 3.12+、`/bin/sh`、`sleep infinity` 和可写 `/tmp`，不要求安装 Iris。Linux Engine 使用宿主 UID:GID，Docker Desktop 使用 `1000:1000`。`HOME=/tmp`、`PYTHONUSERBASE=/tmp/.local`，用户 bin 前置 PATH；环境由镜像、运行默认和显式 environment 覆盖组成，不复制宿主环境。依赖可在用户目录安装，不设置 `PIP_USER`。网络和 CPU/内存/PID 限额固定为整个容器的共享配置。

标准库助手在自己的命令组外管理 `/bin/sh -c`；单命令超时只终止该组，其他命令和服务继续。前台结果以独立的 reason/returncode 回传，用户退出 124/137 不会被猜成超时。输出规则与 Native 相同；读取结果后的临时文件删除失败不改写已知退出结果。

本调用自己的环境清理无法确认时，抛出 `IrisExecutionCleanupError`。若前台结果已知，独立 `command_outcome` 属性保留结果；明确尚未发出命令时，context 中保留 `started=False`。调用者不能把清理失败当作普通未启动错误后宣告资源已经停止。该属性是进程内事实，不写入通用错误详情；后续工具/runtime 接入负责先提交已知工具结果，再交给 run owner 等待清理。

`stop(scope)` 停止整个共享容器。同 session 的调用得到 cancelled，其他 session 正在运行的命令得到 environment_interrupted；不会自动重放命令。新调用等本轮物理停止和旧调用收尾后才重启同一容器，文件和用户依赖保留，后台服务不会自动恢复。无法确认停止时保持入口不可用，显式清理可重试。`aclose()` 停止并删除本服务容器、关闭客户端，保留宿主挂载文件；不扫描或回收用户其他容器。

真实引擎测试默认关闭，显式运行：

```console
uv run --extra sandbox pytest tests/execution/test_docker_integration.py --run-docker -p no:cacheprovider --basetemp=tmp/pytest-docker-local
```

目前 Docker Desktop Linux engine 的实测覆盖同容器复用、中文 cwd 写回、真实退出码与局部超时、共享中断/重启、离线依赖与后台服务复用、只读挂载、有限输出、close 删除与宿主文件保留。已读取真实 cgroup 限额和 none 网络接口状态；资源负载约束与 bridge 网络组合验收仍在交付阶段完成。非 Desktop 的 Linux Engine 尚未实测。
