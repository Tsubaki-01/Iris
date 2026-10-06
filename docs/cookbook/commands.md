# 执行命令和 Python

当任务需要运行已有脚本、处理数据或生成文件时，显式启用 `exec.command` 或 `exec.python`。它们借用同一个 root runner 的命令服务；Native 使用宿主环境，Docker 使用本机 Linux 容器。普通 Python 工具函数、MCP 服务和模型调用不会因为选择 Docker 就全部移进容器。

## 先执行不依赖模型的 Native 示例

完成源码环境准备后，在仓库根目录运行：

```shell
uv run python -m examples.command.native
```

[这个示例](../../examples/command/native.py) 用固定 provider 响应驱动真实文件工具和 shell：写入 CSV 与脚本，调用 `exec_command` 生成 JSON，再调用 `read_file` 读取。成功报告的 `output` 是 `{"rows": 3, "total": 12}`，`confirmed_tools` 包含 `exec_command`。生成文件保留在报告显示的独立 `tmp` workspace，便于检查。

固定 provider 是让执行顺序可复现的替身；命令本身真实执行。示例宿主仅为脚本里预定的调用发送批准响应，不代表普通 Agent 默认拥有命令权限。

接入真实 Agent 的关键配置如下，可合并到已经可用的 `agent.yaml`：

```yaml
tools:
  builtin:
    - file.write
    - file.read
    - exec.command
permissions:
  workspace: workspace
  writes: allow
  execute: confirm
command:
  mode: native
  timeout_seconds: 15
```

`writes` 和 `execute` 分别决定文件写入与命令执行。默认命令执行需要确认；CLI 会呈现交互，自己的宿主按[HITL 指南](hitl-recovery.md)提交 typed response。

Native 在 Windows 使用 `cmd.exe`，其他受支持宿主使用 `/bin/sh`。每次调用是新 shell，前一次 `cd` 或环境变量赋值不会自动进入下一次；需要目录时显式传 `cwd`。命令以宿主用户权限运行，workspace 决定允许的起始目录，不提供操作系统级文件访问隔离。

## 直接执行完整 Python 代码

配置 `exec.python` 后，模型使用 `run_python`，参数是完整 `code`，可选 `cwd` 和 `timeout_seconds`。例如模型调用体：

```json
{"code": "import json\nfrom pathlib import Path\nreport = {'total': sum([3, 4, 5])}\nPath('report.json').write_text(json.dumps(report), encoding='utf-8')\nprint(report)", "cwd": "."}
```

Native 使用 Iris 当前进程的 Python 解释器，Docker 使用镜像中的 Python。每次是独立进程，变量和 import 状态不跨调用保留；工作文件与环境中已安装的依赖可以保留。代码应通过 `print` 输出结果，末尾表达式不会像 notebook 一样自动显示。工具不会自动安装依赖，也不提供交互 stdin 或 TTY。

为了直接观察 Python 执行与产物发布，在仓库根目录保存 `python_report_demo.py`。它复用示例中的固定响应辅助函数，不连接主模型服务：

```python
"""通过真实 Runner 执行 Python 并发布报告。"""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from examples.command._scripted import (
    ScriptedProvider, approve_commands, config_for_workspace,
    done, require_completed, tool,
)
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest


async def main() -> None:
    """创建独立工作区，运行命令并输出发布副本。"""
    workspace = (Path("tmp") / f"python-report-{uuid4().hex[:8]}").resolve()
    workspace.mkdir(parents=True)
    provider = ScriptedProvider()
    provider.steps.extend([
        tool("compute", "run_python", code=(
            "import json\nfrom pathlib import Path\n"
            "report = {'rows': 3, 'total': sum([3, 4, 5])}\n"
            "Path('report.json').write_text(json.dumps(report), encoding='utf-8')\n"
            "print(json.dumps(report))\n"
        )),
        tool("publish", "publish_artifact", file_path="report.json"),
        done("报告已生成并发布。"),
    ])
    runner = AgentRunner.from_config(
        config_for_workspace("python.yaml", workspace), provider=provider,
    )
    try:
        result = await runner.start(AgentRunRequest(input="生成汇总报告。"))
        result = await approve_commands(runner, result, [])
        require_completed(runner, result)
        artifact = runner.list_tool_calls(result.run.run_id)[-1].result.artifact
        print(artifact.path)
        print(json.loads(artifact.path.read_text(encoding="utf-8")))
    finally:
        await runner.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

执行 `uv run python python_report_demo.py`。成功时打印 `.iris/tool-results/` 下的独立副本路径，以及 `{'rows': 3, 'total': 12}`。调用仍经过 runner、权限确认和工具结果提交；固定 provider 只决定调用顺序，不执行 Python 计算。

## 使用本地 Docker 环境

Docker 是可选依赖，需要本地 Linux engine 和预先准备的镜像。仓库提供 [Dockerfile](../../Dockerfile)。在仓库根目录按以下顺序操作：

```shell
uv sync --extra sandbox
docker build --load -t iris-command:local .
uv run --extra sandbox python -m examples.command.docker --run-docker
```

Iris 的 prepare 只使用已有引擎与镜像，不会自动 build 或 pull。若自备镜像，可用示例的 `--image` 参数，镜像须提供实现所需的 Linux shell 与 Python 环境。完整条件以 [Docker 后端](../../src/iris/command/docker.py)及 [Docker 示例](../../examples/command/docker.py)为准。

一个 root runner 的所有 session 和 child 共用容器与 `/workspace` 挂载。默认网络为 `none`，CPU、内存、进程上限在 `command.docker` 配置；参数默认值见[参考](../reference/tools.md#命令环境)。示例验证两个 session 复用后台服务、一次 run 失败后停止共享环境、后续显式重启服务。成功报告包含 `waiting_survived: true`，停止后 `service_running: false`，重启后 `total: 12`。

child 不能另行声明 `command`；它借用 root 已确定的模式和服务。child 的 workspace 只确定默认 cwd，`writes: deny` 限制内置文件写工具，不会把 root 可写挂载中的 Docker 命令变成只读。Native 的只读 workspace 配置不能同时注册命令工具。

## 期限、失败与关闭

最终命令期限取 root 默认、调用传入期限和 runtime 工具期限中最短者。单命令超时停止本次执行；普通非零退出码返回 `COMMAND_FAILED` 工具错误，模型仍可以继续解释或修正，不必让整个 run 失败。

run 异常退出时，harness 要先停止相应命令资源并等待旧调用排空，再提交终态。Docker 的这一操作影响 root 共享环境，因而可能连带中断其他命令；WAITING 的会话状态和工作文件不会因此被删除。正常完成或等待人工输入会保留命令环境，结束使用时调用 `await runner.aclose()`，由 root 关闭它拥有的资源；Docker 容器此时删除，宿主 workspace 文件保留。

命令结果记录退出状态、stdout/stderr、耗时及输出采集统计。采集端达到额度时只保留头尾，丢弃的字节不能靠 artifact 找回；之后的工具正文截断则会把已经采集到的完整正文保存为 artifact。这两个限制需要区分。

后续阅读：[Skill、委派与共享资源设计](../design/delegation.md)、[命令工具参考](../reference/tools.md#命令环境)、[运行与关闭 SDK](../reference/runtime.md)。源码入口：[命令协议](../../src/iris/command/service.py)、[公共命令工具入口](../../src/iris/tools/builtin/_command.py)、[harness 命令生命周期](../../src/iris/harness/_command_lifecycle.py)。
