# 提供任务背景和每步动态上下文

给模型的材料有不同更新节奏：系统规则通常稳定，一次任务的背景随输入保存，界面中的当前状态可能每步变化。正确选择入口，可以避免重复追加背景或把旧状态误当成现在。

## 固定背景使用 context.yaml

先按[配置配方](configure-agent.md)使用 `context.path`。一个完整 `context.yaml` 可以是：

```yaml
system:
  slots:
    - name: identity
      order: 10
      content: 你是项目分析助手，依据材料作答。
memory:
  slots:
    - name: project_background
      content: 项目面向 Python 开发者，重点关注本地使用和应用集成。
before_current_input:
  slots:
    - name: response_preference
      content: 解释时先给结论，再给依据；不熟悉的概念先定义。
```

`system` 形成系统消息；`memory` 是固定背景槽位，名字不代表它会自动查询长期记忆数据库；`before_current_input` 是本次输入的前置背景，会随输入保存，恢复时不会重复追加。

同一 section 的 slot 按 `order`，再按 `name` 排序。值越小越靠前。默认渲染保留结构，也可以设置 section 的 Jinja `template`；具体格式、字符上限与模板路径见[Context 参考](../reference/context.md)。

## 当前状态通过 ContextSource 提供

假设你的宿主有“当前打开的文件”。它不是用户的新指令，也不适合每次改变就追加一条永久历史。可以在每个模型步骤前返回一份完整快照。

在快速开始的仓库根目录保存 `run_with_context.py`，使用已有 `agent.yaml` 和凭据：

```python
"""向每个模型步骤提供宿主当前状态。"""

import asyncio
from dataclasses import dataclass

from iris import init_config
from iris.context import ContextBuildScope, ContextContribution, ContextSnapshot
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest


@dataclass
class EditorContext:
    """由界面更新的当前文档状态。"""

    active_document: str

    async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
        """返回本步骤的全部状态，不继承上一步遗漏的条目。"""
        return ContextSnapshot(
            contributions=(
                ContextContribution(
                    key="active-document",
                    text=f"当前打开：{self.active_document}",
                ),
                ContextContribution(
                    key="display-note",
                    text=f"宿主正在处理会话 {scope.session_id}。",
                    required=False,
                    priority=10,
                ),
            )
        )


async def main() -> None:
    """将动态来源绑定到 Runner，再提交一次输入。"""
    init_config()
    source = EditorContext(active_document="pyproject.toml")
    runner = AgentRunner.from_config_path("agent.yaml", context_source=source)
    try:
        result = await runner.start(
            AgentRunRequest(input="我当前打开的是哪个文件？", session_id="context-demo")
        )
        if result.assistant_message is not None:
            print(result.assistant_message.text)
        print(result.run.phase.value, result.run.stop_reason)
    finally:
        await runner.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

```powershell
uv run python run_with_context.py
```

这份快照告诉模型当前文件名，不会读取文件正文。需要正文时仍要调用工具。实际界面可以更新 `source.active_document`，下一次采集采用新值；多个会话共享一个 source 时，应按 `scope.session_id` 提供对应会话的状态。

`required=True` 的内容必须保留；可选内容在上下文紧张时可被移除，较低 `priority` 先退出。框架不会替你判断哪些用户约束可以丢弃。

每个获准的主模型步骤采集一次，工具等待恢复不为了同一步再次采集。快照替换此前动态状态，不直接进入会话历史；需要长期追溯的材料应由宿主另行保存。

## 为长对话设置输入预算

在 Agent YAML 中增加下列片段：

```yaml
compaction:
  input_budget_tokens: 32000
  keep_recent_ratio: 0.15
  summary_ratio: 0.05
context_policy:
  enabled: true
  preserve_recent_tool_groups: 2
  old_result_preview_chars: 512
```

`input_budget_tokens` 是你已为模型输出预留空间后的可用输入额度，不会根据 `model.max_tokens` 自动再减一次。应按实际模型窗口设置，不能直接复制一个超过模型能力的数字。

系统要求、历史、快照、工具定义和输出格式共用此额度。完整输入达到固定 80% 压力线后，Iris 依次尝试可用的确定性减载，再判断是否摘要旧历史。不是每一步都会调用摘要模型，也不是仅计算聊天文本长度。

压缩不会删除原始历史。CLI 会报告压缩开始、完成或未完成；一次已经开始但失败的压缩会结束当前 Run，不继续发送未达要求的候选请求。成功条件和原文回读见[上下文工程总览](../design/context-engineering.md)。

需要查回先前材料时，可以要求 Agent“先搜索当前会话中关于依赖版本的记录，再读取命中的原文”。启用 context policy 后，模型可用 `context_search` 找到 `message:N` 或 `result:N:B` 引用，再用 `context_read` 按字符分页读取。搜索只覆盖已保存正文和预览；空命中但仍有下一页时还可继续。参数、分页游标和原生 artifact 读取规则见[回读工具参考](../reference/context.md#回读与搜索工具)。

## 选择合适的入口

| 材料 | 入口 |
| --- | --- |
| Agent 长期遵循的规则 | `system` 或 context 的 system slots |
| 人工维护的固定项目背景 | context 的 memory slots |
| 这次任务开始时的背景 | before_current_input |
| 每步可能改变的界面或工作状态 | `ContextSource.collect` |
| 跨任务积累并按需查询的知识 | [长期记忆](memory.md) |
| 当前会话清单和持续目标 | [Todo 与 Goal](goals-todos.md) |

下一步：[Context 精确契约](../reference/context.md) · [工具按需披露](../design/tool-execution.md)。
