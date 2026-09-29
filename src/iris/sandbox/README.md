# 本地 Docker 沙箱资源

`iris.sandbox` 管理本地 Docker 引擎连接和一个容器的资源。命令执行入口仍由
[`iris.command`](../command/README.md) 提供；普通文件、Memory 和其他工具继续在宿主运行。

```python
from iris.command import CommandConfig
from iris.sandbox import DockerConfig

config = CommandConfig(
    mode="docker",
    docker=DockerConfig(image="iris-command:local", network="none"),
)
```

YAML 仍使用 `command.mode`、`command.timeout_seconds` 与 `command.docker`，配置示例见
[`iris.agents`](../agents/README.md#commandconfig)。`DockerConfig` 的公开导入位置是
`iris.sandbox`。导入和解析配置不连接 Docker，不加载可选驱动。

## 准备镜像

先准备 Linux Docker Engine 或 Docker Desktop 的 Linux engine，再安装 extra 并在仓库根目录构建：

```console
uv sync --extra sandbox
docker build --load -t iris-command:local .
```

根目录 [Dockerfile](../../../Dockerfile) 使用 `python:3.12-slim`，不预装项目无关的工具。
[.dockerignore](../../../.dockerignore) 排除本地环境、缓存和私有文档等无关构建内容。
`--load` 将镜像载入本地引擎；首次获取基础镜像或安装依赖可能联网，运行时 `network: none`
不限制开发者显式执行的 build。Iris 不自动安装 Docker、构建或拉取镜像，也不回退宿主执行。

需要额外依赖时，可编辑 Dockerfile 后重新构建，例如：

```dockerfile
FROM python:3.12-slim
RUN python -m pip install --no-cache-dir httpx
```

使用 `docker build --load -t my-agent-command:local .` 构建后，将 `command.docker.image`
设为 `my-agent-command:local`。仅安装 Iris wheel 的开发者也可在自己的目录保存该配方构建。
依赖应全局安装，或放在所有运行用户可读的目录（如 `/opt/agent-tools`）；不要安装到
`/root/.local`，也不要放在会被项目挂载覆盖的 `/workspace`。

镜像需提供 Python 3.12+、`/bin/sh`、`sleep infinity` 和可写 `/tmp`。命令 helper 与 Python
loader 仍由 command 包携带并在调用时传入，镜像不需要安装 Iris。

## 配置与容器行为

`DockerConfig` 默认使用 `iris-command:local`、`none` 网络、2 CPU、1024 MiB 内存及 128 PID。
`endpoint` 仅接受本地 Unix socket 或 Windows named pipe；默认分别是
`unix:///var/run/docker.sock` 和 `npipe:////./pipe/docker_engine`，不读取 Docker CLI context
或 `DOCKER_HOST`。构建必须使用同一本地引擎，必要时显式执行
`docker --host ENDPOINT build --load -t iris-command:local .`。

运行时决定挂载、运行用户、工作目录、常驻启动命令、网络和资源限额。root 目录单一挂载到
`/workspace`，只读与否由 root 的 `workspace_writable` 决定。Linux Engine 使用宿主 UID:GID，
Docker Desktop 使用 `1000:1000`。`HOME=/tmp`、`PYTHONUSERBASE=/tmp/.local`，用户 bin
前置 PATH；环境由镜像、运行默认和显式 `environment` 覆盖组成，不复制宿主环境。

一个 root 命令服务拥有一个容器，session/child 共享其文件、依赖、后台服务和资源额度。
正常命令退出不停止容器；stop/start 复用原容器，保留文件和用户依赖，但不自动恢复后台服务。
重建镜像 tag 不替换已有容器；使用新镜像前先关闭旧 root，再创建 runner。
最终关闭只删除本实例容器并关闭客户端，保留宿主挂载文件。

## 实现边界

`DockerSandbox` 位于 `sandbox/docker.py`，供 Docker 命令适配层使用：

- `prepare()` 准备驱动、检查 Linux 引擎与现有镜像，不创建容器。
- `create()`、`start()` 分开执行物理操作，保留命令层在操作之间响应停止的机会。
- `stop()` 确认本容器已停止；`aclose()` 删除已停止的容器并关闭客户端。

资源对象拥有 client/container、运行用户和资源状态，不拥有 session、call、停止收据或调度锁。
命令服务串行协调资源控制，先停止并排空命令，再关闭资源；不应绕过命令服务与在途执行并发
操作资源对象。命令层继续管理调用准入、单命令期限、输出、known/unknown 和两阶段停止。

底层资源失败抛出 `IrisSandboxError`；command 在准备、执行、停止或关闭边界转换为相应命令
错误并保留已有结果事实。沙箱错误不携带命令结果，不反向依赖 command 模型。
