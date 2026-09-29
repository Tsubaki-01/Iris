# 命令执行

本包提供 Native 与可选本地 Docker 命令环境。两种后端可直接通过服务调用，也可由 Agent 显式注册 `exec_command` 或 `run_python` 使用；普通工具仍在宿主执行。

```python
from iris.command import CommandConfig

native = CommandConfig()  # 默认 120 秒
docker = CommandConfig.model_validate({"mode": "docker"})
```

`DockerConfig` 默认使用显式构建的 `iris-command:local` 镜像、`none` 网络、2 CPU、1024 MiB 内存和 128 PID。`endpoint` 只接受本地 Unix socket 或 Windows named pipe。配置解析和导入本包都不连接 Docker；Native 模式显式声明 `docker` 块会报错。

`CommandService.execute(scope, request)` 接收已解析的宿主 cwd、最终命令期限和 typed payload：`ShellCommand(command)` 或 `PythonCode(code)`，返回前台执行事实。两种载荷共用同一服务的停止与排空，不创建独立 Python 环境。服务不拥有工具权限、生命周期历史或模型调用。

停止分为两个完成点：同步 `stop(scope)` 立即登记并调度操作；工具 body 只等待 `wait_stopped()` 的物理停止证明，外层结算等待 `wait_drained()` 确认旧调用收尾。`CommandStopReceipt` 仅标识一次已证实的停止，不包含 Future 或资源句柄，不持久化。消费旧收据不得再次停止之后重启的环境。

`CommandStopSlot` 是当前调用因果链的可写共享状态，保留 receipt 与尚未完成的 cleanup_error。context 投影保留槽的 identity，避免工具结果被 middleware 替换后丢失清理证明；它不是模型输入或历史数据。

`CommandBinding` 绑定配置、共享服务与 `CommandEnvironment`。root 装配一次，child 借用同一绑定和调用槽，不单独关闭服务。Agent 配置与权限示例见 [agents](../agents/README.md)，工具参数与结果见 [tools](../tools/README.md)。

## 直接调用 Native

```python
import asyncio
from pathlib import Path

from iris.command import CommandRequest, CommandScope, ShellCommand
from iris.command.native import NativeCommandService


async def main() -> None:
    workspace = Path.cwd().resolve()
    service = NativeCommandService(workspace)
    try:
        await service.prepare()
        result = await service.execute(
            CommandScope(run_id="example", session_id="local"),
            CommandRequest(
                call_id="hello", payload=ShellCommand("echo hello"), cwd=workspace, timeout_seconds=5
            ),
        )
        print(result.status, result.exit_code, result.stdout)
    finally:
        await service.aclose()


asyncio.run(main())
```

Windows 使用系统目录中的 `cmd.exe`，POSIX 使用 `/bin/sh`。每次命令创建新 shell，继承宿主环境和 PATH，无交互 stdin、无 TTY，不保留上次的 `cd` 或环境变量修改。Windows 进程隐藏窗口，host 必须使用支持 subprocess 的事件循环；库不更改全局 event loop policy。

Python 载荷使用 `PythonCode("print(1 + 2)")`。Native 直接启动 `sys.executable`，不经过 shell；代码写入调用专属的系统临时文件，正常结束后删除。每次新进程，依赖来自运行 Iris 的同一 Python 环境；变量不延续，工作区文件保留。cwd 是项目模块的默认导入位置，回溯使用 `<iris-python>` 与原代码行；不承诺真实脚本 `__file__`。输出使用 UTF-8 和非缓冲模式，末尾表达式不会自动显示，应使用 `print`。

Native **不是 OS 级隔离沙箱**。直接服务调用接收已解析的宿主目录，不代替工具层的 workspace 和权限裁决。普通退出保留实际退出码，包括 124/137；期限终止使用独立的 `timed_out` 状态。stdout/stderr 各保留最多 512 KiB，各自一半头部、一半最新尾部；未超额时内容完整。持续排空超额数据，以 UTF-8 replacement 解码。前台退出后管道最多再排空 1 秒，后台继承管道时标记可能截断并返回。

`CommandOutcome.output_stats` 记录两条流的实际采集字节数、保留字节数及截断原因：`byte_limit`、`drain_timeout`、`stream_error`、`stream_closed`。`output_truncated` 由原因集合派生。提前结束采集时，计数只表示已读取字节，后续数量未知；被丢弃的原始输出不能从工具结果存档找回。

单命令期限仅停止当前调用。`stop(scope)` 停止该 session 的当前调用，不影响独立 session；`aclose()` 停止所有仍持有的调用并拒绝新执行。POSIX 对本命令组发 TERM、有限等待后 KILL；Windows 用隐藏的 taskkill 尽力终止子树，再确认所持有前台退出。服务不追踪历次命令留下的后台程序，也不承诺回收宿主所有后代。重复取消不会打断已经开始的必要收尾；停止无法确认时抛出公共 unknown 并保留可重试的清理状态。

## 本地 Docker

开发环境先准备 Linux Docker Engine，或 Windows Docker Desktop 的 Linux engine，再从仓库根目录安装 extra 并显式构建镜像：

```console
uv sync --extra sandbox
docker build --load -t iris-command:local .
```

根目录 [Dockerfile](../../../Dockerfile) 仅以 `python:3.12-slim` 为基底，不预装项目无关的工具；[.dockerignore](../../../.dockerignore) 排除本地环境、缓存和私有文档等无关构建内容。`--load` 将构建结果载入本地镜像库。首次获取基础镜像或安装包可能联网；运行时 `network: none` 不限制开发者显式执行的 build。

服务不会安装 Docker、构建或拉取镜像，也不会回退 Native。基础导入不加载 aiodocker；仅 Docker 的 `prepare()` 导入驱动并检查本地引擎、Linux 容器模式和预备镜像。默认 endpoint 在 Windows 为 `npipe:////./pipe/docker_engine`，Linux 为 `unix:///var/run/docker.sock`；显式传给驱动，不由 Docker CLI context 或 `DOCKER_HOST` 改写。因此 build 必须使用该 endpoint 对应的同一本地 engine；CLI 使用其他 context 时，切换回对应引擎或用 `docker --host ENDPOINT build --load -t iris-command:local .` 指定它。

按需编辑 Dockerfile 即可预装依赖。例如需要 HTTP 客户端时，可采用：

```dockerfile
FROM python:3.12-slim
RUN python -m pip install --no-cache-dir httpx
```

用 `docker build --load -t my-agent-command:local .` 构建后，将 `command.docker.image` 设为 `my-agent-command:local`。只有已安装 Iris wheel 的开发者，也可在自己的目录保存上述配方再构建。预装依赖应全局安装，或通过 `COPY` 放入所有运行用户可读的目录（如 `/opt/agent-tools`）；不要安装到 `/root/.local`，也不要放在会被项目挂载覆盖的 `/workspace`。命令 helper 保留在 Iris 包中，执行时传入容器，不需要把 Iris 安装到镜像里。

Dockerfile 只声明依赖环境；runtime 继续决定 root 挂载、运行用户、工作目录、常驻启动命令、网络和资源限额。已有容器不会随镜像 tag 重建而替换：使用新镜像前先关闭旧 root，再创建新 runner；普通 stop/start 仍复用原容器。

在前面的调用示例中，可将服务构造替换为：

```python
from iris.command import DockerConfig
from iris.command.docker import DockerCommandService

service = DockerCommandService(workspace, DockerConfig(), workspace_writable=True)
```

一个服务只拥有一个容器，同服务的 session 与 child scope 共用文件、依赖、后台服务和资源额度；两个服务各自独立。首次实际命令才创建和启动容器，正常退出不会停容器。root 目录单一挂载到 `/workspace`，child cwd 相对 root 投影；命令可以访问整个 root 挂载，不获得每个 session/child 独立的目录隔离。挂载只读由 `workspace_writable` 决定，授权模型命令不等于逐次拦截 shell 内部文件操作。

镜像需提供 Python 3.12+、`/bin/sh`、`sleep infinity` 和可写 `/tmp`，不要求安装 Iris。Linux Engine 使用宿主 UID:GID，Docker Desktop 使用 `1000:1000`。`HOME=/tmp`、`PYTHONUSERBASE=/tmp/.local`，用户 bin 前置 PATH；环境由镜像、运行默认和显式 environment 覆盖组成，不复制宿主环境。依赖可在用户目录安装，不设置 `PIP_USER`。网络和 CPU/内存/PID 限额固定为整个容器的共享配置。

标准库助手在自己的命令组外管理 `/bin/sh -c` 或镜像内的 Python 进程；单命令超时只终止该组，其他命令和服务继续。Python 源码在启动前上传到容器 `/tmp`，因此只读 workspace 挂载仍能执行不写工作区的代码。两后端使用同源 Python 启动器，不要求镜像安装 Iris，也不自动安装用户代码依赖。

前台结果以独立的 reason/returncode 回传，用户退出 124/137 不会被猜成超时。输出规则与 Native 相同；读取结果后的临时文件删除失败不改写已知退出结果。容器停止后不会为了删除本次临时源码而自动重启，残留随服务最终删除容器一起清理。

本调用自己的环境清理无法确认时，抛出 `IrisCommandCleanupError`。若前台结果已知，独立 `command_outcome` 属性保留结果；明确尚未发出命令时，context 中保留 `started=False`。调用者不能把清理失败当作普通未启动错误后宣告资源已经停止。该属性是进程内事实，不写入通用错误详情；工具/runtime 先提交已知工具结果，再交给 run owner 等待清理。Harness 的 pending、deadline 与 child 交接见 [运行结算](../harness/README.md#cancellation-与-recovery)。

`stop(scope)` 停止整个共享容器。同 session 的调用得到 cancelled，其他 session 正在运行的命令得到 environment_interrupted；不会自动重放命令。新调用等本轮物理停止和旧调用收尾后才重启同一容器，文件和用户依赖保留，后台服务不会自动恢复。无法确认停止时保持入口不可用，显式清理可重试。`aclose()` 停止并删除本服务容器、关闭客户端，保留宿主挂载文件；不扫描或回收用户其他容器。

真实引擎测试默认关闭，显式运行：

```console
uv run --extra sandbox pytest tests/command/test_docker_integration.py --run-docker -p no:cacheprovider --basetemp=tmp/pytest-docker-local
```
