# 本地命令执行示例

两个示例都由固定响应的 `ScriptedProvider` 驱动，经过真实 `AgentRunner`、权限交互与
`exec_command`，不需要模型 API key。文件读写仍使用原生文件工具。以下命令从仓库根目录运行。

## Native：CSV → 命令 → JSON

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run python -X utf8 -m examples.execution.native
```

[native.yaml](native.yaml) 显式注册 `exec.command`。脚本先调用 `write_file` 写入 `input.csv`
和 [transform.py](transform.py)，再用当前 Python 解释器执行转换，最后调用 `read_file` 读取
`output.json`。输出包含实际 shell（Windows 为 `cmd.exe`，Linux 为 `/bin/sh`）、工具调用次序、
权限确认记录，以及 `{"rows": 3, "total": 12}`。

Native 使用宿主用户权限；workspace/cwd 是文件工具范围与命令起始位置，不是命令的 OS 沙箱。
基础安装即可运行，不需要 Docker extra，也不连接 Docker。

## Docker：同 root 复用与停止后重启

先自行准备本机 Linux engine 与 `python:3.12-slim` 镜像，例如在需要联网准备的阶段执行
`docker pull python:3.12-slim`。示例本身不会 pull/build、联网安装依赖或访问外部网络。
已有其他包含 Python 标准库的 Linux 镜像时，可以传入 `--image IMAGE`。

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run --extra sandbox python -X utf8 -m examples.execution.docker --run-docker
```

没有 `--run-docker` 时，CLI 直接提示显式启用，不探测引擎。
[docker.yaml](docker.yaml) 设置 `network: none`、1 CPU、256 MiB 内存、64 PID 和 15 秒单命令期限。
容器内的 loopback 服务不需要外网或端口发布。脚本执行以下流程：

1. Session A 用原生文件工具写入 [offline_stats.py](offline_stats.py) 和服务脚本；命令把离线
   模块复制到容器 `/tmp`，启动受控 HTTP 后台服务。A 正常完成后服务继续运行。
2. 同一个 root 的 session B 请求服务并通过原生文件工具读取结果，然后停在问题交互。
3. A 的下一次 run 模拟 provider 失败。Runner 先停止共享容器，再提交 FAILED；B 仍保持 WAITING。
4. 宿主回答 B 的问题。下一条命令重启环境，确认容器文件层中的模块仍存在、后台服务已经停止。
   B 显式重新启动服务，再次请求到 `{"total": 12}`。

容器共享范围是这个 root 实例，独立 root 不共享。每次命令使用新 shell，`cd`/`export` 不延续；
需要持续复用的模块或环境应写入文件。后台服务显式脱离前台并关闭标准流，单命令超时只处理
当前命令；run 失败或取消会停止 root 的共享命令环境。停止不回滚工作文件，也不自动恢复服务。
root `aclose()` 删除本次容器，宿主 workspace 内的文件保留。修改 execution 配置需关闭并新建 root。

## 确认与 child 权限

两个 root YAML 都写出了 `execute: confirm`；省略该字段时也是这个默认值。示例宿主只对代码中
预先声明的固定命令发送 `PermissionInteractionResponse(decision="approve")`，报告里会列出每次确认。
这不改变普通 Agent 的权限策略。将 YAML 交给真实模型时，需要另行配置 provider 凭据并由宿主
展示/处理 HITL。

[child.yaml](child.yaml) 是可放入 Docker root subagent catalog 的完整 child 配置。它不声明
`execution`，借用 root 的 mode、服务、文件层和停止范围。`writes: deny` 仅限制原生文件工具；
`execute: confirm` 仍允许经确认的命令，而命令的实际写入能力取决于 root `/workspace` 挂载权限。
child workspace 只指定默认 cwd，不能把这个 child 称为“整体只读 Agent”。Native 不接受
`writes: deny` 与 exec 同时启用的配置。

## 运行与验证

脚本默认在 `tmp/` 下为本次运行创建带中文和空格的独立 workspace，并打印绝对路径；文件保留供查看。
也可传 `--workspace PATH` 指定一个新的空目录，避免覆盖自己的文件。

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run pytest tests/examples/test_execution_examples.py -p no:cacheprovider --basetemp="$PWD\tmp\pytest-execution-examples"
```

默认验证配置与真实 Native 工具闭环，Docker 用例跳过且不连接引擎。真实 Docker 测试使用本文件局部的
环境变量开关，不依赖 `tests/execution` 的 `--run-docker` 选项：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
$env:IRIS_RUN_DOCKER_EXAMPLES = "1"
uv run --extra sandbox pytest tests/examples/test_execution_examples.py -p no:cacheprovider --basetemp="$PWD\tmp\pytest-execution-examples-docker"
Remove-Item Env:IRIS_RUN_DOCKER_EXAMPLES
```

显式启用后，缺少 extra、引擎或预备镜像会报错，不会被记成通过。示例验证文件与生命周期语义；
资源约束、跨平台进程行为的完整实测位于 `tests/execution/`，不由这两个教学脚本替代。
