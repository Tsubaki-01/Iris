# 将 Agent 接入 Python 应用

沿用[快速开始](quickstart.md)里的 `agent.yaml` 和凭据，本页用 Python 接管输入与输出。Agent 的声明保持不变，宿主从 CLI 换成你的程序。

## 一次完整调用

在仓库根目录、`agent.yaml` 旁保存 `run_agent.py`：

```python
"""用 Python 宿主完成一次 Iris 调用。"""

import asyncio

from iris import init_config
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest


async def main() -> None:
    """初始化应用，提交任务并关闭所拥有的资源。"""
    init_config()
    runner = AgentRunner.from_config_path("agent.yaml")
    try:
        result = await runner.start(
            AgentRunRequest(
                input="请解释 Python 的异步迭代器适合什么场景。",
                session_id="sdk-demo",
            )
        )
        print("运行阶段：", result.run.phase.value)
        print("停止原因：", result.run.stop_reason)
        if result.assistant_message is not None:
            print(result.assistant_message.text)
        if result.pending_interaction is not None:
            print("等待人工输入：", result.pending_interaction)
        if result.error is not None:
            print("运行错误：", result.error.message)
    finally:
        await runner.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

执行：

```powershell
uv run python run_agent.py
```

`init_config()` 读取当前环境变量，每个进程初始化一次；使用 dotenv 时改为 `init_config(env_file=".env.local")`。Runner 装配模型、工具和存储，`start()` 返回到达等待或终态的 `RunResult`。模型回答在 `assistant_message.text`，运行状态和用量在 `result.run`。

不要把“方法返回了”直接当成“任务成功”：工具可能请求人工输入，Run 也可能因预算、取消或失败停止。本例把不同结果直接打印，交互应用应按[人工交互与恢复指南](../cookbook/hitl-recovery.md)继续处理。

`finally` 对应宿主的资源所有权。构造时注入的共享服务由其创建者管理；Runner 自己创建的资源由 Runner 收尾。需要长期服务时，可以复用 Runner，不必为每条消息重新构造。

## 连续交谈与运行中输入

顺序调用 `runner.start()`，请求使用相同 `session_id`，即可接续同一个会话的历史。上一次 Run 仍在等待时，应提交它需要的 typed response；不要用另一个 `start()` 冒充回答。

若界面允许用户在模型执行期间输入，使用 `SessionManager` 组织普通输入、steer、follow-up 与 interrupt。它处理单会话的进程内输入顺序，Runner 仍负责持久运行状态。完整接线见[管理会话](../cookbook/sessions.md)。

## 让配置与宿主保持各自职责

| 适合放 YAML | 适合由 Python 宿主提供 |
| --- | --- |
| 模型、系统要求、工具、workspace、持久化选项 | 用户输入、界面、服务启动与关闭 |
| 是否启用 Skill、记忆、Goal 等能力 | 共享维护协调器、需要借用的服务实例 |
| 静态 context 与扩展引用 | 每步动态状态 `ContextSource`、流式事件接收方 |

有现成 typed 配置时也可以使用 `AgentRunner.from_config(config, config_path=...)`。传入 `config_path` 是为了保留相对路径语义，不需要再次读取同一 YAML。完整注入参数见[运行 SDK 参考](../reference/runtime.md)。

接下来按宿主需求选择：[动态上下文](../cookbook/context.md) · [流式界面](../cookbook/streaming.md) · [图片与语音](../cookbook/media.md) · [后台记忆维护](../cookbook/memory.md)

实现入口：[AgentRunner](../../src/iris/harness/runner.py)、[请求与结果模型](../../src/iris/lifecycle/models.py)、[进程配置](../../src/iris/config.py)。
