# 启用长期记忆

长期记忆适合保存跨会话仍有用的事实、偏好和约定。它与当前会话消息分开存储；启用后，模型根据概览决定何时搜索，需要完整条目时再读取。本文先给出 CLI 的自动维护配置，再用不调用模型的 SDK 示例说明如何检查、修改真实条目。

前提：已完成[快速开始](../getting-started/quickstart.md)，模型连接可用，工作目录可写。配置字段的完整含义见[长期能力参考](../reference/memory-goals.md)。

## 让 Agent 保存和使用记忆

在工作目录保存 `agent.yaml`：

```yaml
name: project-assistant
model: deepseek/deepseek-flash
system: |
  帮助用户处理项目工作。需要历史约定时先查看记忆概览，按需搜索。
  只有用户明确要求保存的信息才使用记忆写工具，并说明保存原因。
permissions:
  workspace: .
memory:
  enabled: true
  read_namespaces: [project]
  write_namespace: project
  generation:
    enabled: true
maintenance:
  idle_seconds: 300
  min_pending_runs: 10
tools:
  builtin:
    - memory.remember
    - memory.update
    - memory.forget
```

在该目录启动：

```powershell
uv run iris chat agent.yaml
```

可以先输入“请记住：本项目的文档示例统一用中文解释，代码标识符保留英文”，再询问已有约定。这里有两条不同的写入路径：

- **显式写入**：模型调用 `memory_remember` 后，正式条目立即进入数据库。`memory_update` 修改同一条目，`memory_forget` 软删除条目。
- **自动整理**：已提交的本轮材料先成为 Episode；宿主空闲后从材料提炼 Observation，再整理成正式 Item，并刷新概览。生成可能决定没有值得新增的内容，因此不能以“必须生成一条记忆”作为成功标准。

`memory_search` 和 `memory_fetch` 在 `memory.enabled: true` 时自动注册，不要把 `memory.search`、`memory.fetch` 写入 `tools.builtin`。三个写工具需要上面的显式声明。自动整理也可以关闭，此时仍能用读写工具及 SDK 管理记忆。

CLI 会装配维护协调器。自动开始一批原文学习，默认同时要求宿主持续空闲五分钟，以及当前
数据库/namespace 积累十个合格新 Run：已终态、完整捕获、有有效原文且会话没有等待中的人工交互。
一个 Run 捕获多页只计一次；捕获落盘仍及时进行。退出 CLI 后不会留一个脱离宿主的定时服务。

获准的一批按预算分轮处理，重启后余料也不用重新凑十个，新 Run 则另行累计。旧 Observation、
显式记忆变更和投影/概览修复不用等待十个新 Run。不足数量时可以一直等待，时间再久也不会
自动放行；希望更及时自动整理时，把 min_pending_runs 设为 1，或由 SDK 宿主显式请求整理。

在同一会话里马上追问约定，回答也可能只是来自聊天历史。要体验跨会话记忆，应先确认概览已经发布，再用新的 `--session-id` 启动会话并询问概览覆盖的约定。数据库条目、概览文件和模型实际读取是不同的观察位置，下节分别说明。

## 检查真正保存了什么

默认数据库位于 workspace 的 `.iris/memory/memory.db`。下面的完整脚本在该目录保存为 `inspect_memory.py`，不需要模型请求；它显式写入一条用于演示的记忆、搜索并读取完整条目。

```python
from pathlib import Path

from iris.memory import (
    FileMemoryMirror,
    MemoryItemPatch,
    MemorySearchQuery,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
)

workspace = Path.cwd()
mirror = FileMemoryMirror(workspace / ".iris/memory", workspace_root=workspace)
mirror.initialize_layout()
service = MemoryService(
    SQLiteMemoryStore(workspace / ".iris/memory/memory.db"),
    mirror=mirror,
)

item = service.remember(
    MemoryWriteInput(
        text="文档示例统一用中文解释，代码标识符保留英文。",
        reason="用户明确指定文档约定",
        category="reference",
    )
)
hits = service.search(MemorySearchQuery(query="文档"), namespaces=["project"])
for hit in hits.items:
    print(hit.item_id, hit.snippet, hit.is_complete)

current = service.get_item(item.id, namespaces=["project"])
print(current.text if current is not None else "未找到活跃条目")

updated = service.update(
    item.id,
    "project",
    MemoryItemPatch(text="文档正文和示例解释使用中文，代码标识符保留英文。"),
    reason="补充约定适用范围",
)
print(updated.text)
print(service.generation_state("project"))
```

```powershell
uv run python inspect_memory.py
```

SDK 写入会得到稳定 `item.id`，后续修改复用这个 ID。重复执行脚本会新增条目，实际业务应保存 ID 后按需更新。`generation_state` 显示待处理材料、观察、显式变更，以及当前条目和投影版本；它不会启动维护。

分类 Markdown 位于 `.iris/memory/namespaces/<namespace key>/`，可供人工阅读。`Memory.md` 是模型生成的概览，包含“核心事实”和“可查询的知识”；分类正文是数据库的派生投影。修改这些文件不会更新权威条目，正式修改使用 SDK 或记忆写工具。

## 新记忆什么时候进入模型请求

新会话首次输入时，Iris 采用当前已发布概览；成功压缩对话时再采用一次。普通后续 Run、HITL resume 和 checkpoint recover 使用已保存窗口，不会每次重新生成或替换概览。

这意味着“数据库已更新”与“当前会话概览已更新”是两件事。对于概览已覆盖的主题，Search/Fetch 读取当前活跃条目，即使概览较旧，也能获取更新后的数据库内容。默认 memory 提示要求不查询概览未覆盖的主题；没有概览时正常聊天，但本窗口暂不查询长期记忆。读工具仍已注册，不代表模型已经获得知识范围，也不会因此隐式触发一次概览生成。

全部读取 namespace 共用概览额度，默认是输入预算的 2%。装不下完整概览时，Iris 尝试只放完整知识范围；知识范围仍超额会报告上下文错误。调整概览规模或预算字段后重建 runner，相关规则见[参考](../reference/memory-goals.md)。

## 在 Python 宿主中维护

如果 SDK 配置开启 `memory.generation.enabled`，需要显式构建共享 `MemoryService`，并在运行前用 `runner.bind_maintenance()` 绑定 `MaintenanceCoordinator`。仅调用 `AgentRunner.from_config_path()` 不会替宿主建立后台维护调度。完整装配和关闭顺序见[经验与维护配方](evolution.md#python-宿主的完整装配)。

宿主调用 `await coordinator.request_memory_cycle(binding)` 可提前执行一轮，跳过自动的时间和
数量门槛，仍等待前台退出并遵守来源、WAITING 和锁约束。`snapshot()` 中 waiting_for_materials
配合 pending_new_runs/min_pending_runs 可显示例如 7/10；它是最近一次判定的投影，不会同步读库。

只想手动运行生成阶段时，可为服务提供 provider、model 和 `PromptSource`，依次调用 `await service.flush("project")`、`await service.dream("project")`、`await service.refresh_overview("project")`。这些是独立模型请求；查看各自结果和 usage，不把它们算作主 Run 的生成质量或成本。

下一步：阅读[记忆设计](../design/memory.md)，理解材料、知识、投影与模型视图的分工；精确接口见[长期能力参考](../reference/memory-goals.md)。

实现入口：[记忆服务](../../src/iris/memory/service.py)、[工具装配](../../src/iris/agents/config/tools.py)、[窗口采用](../../src/iris/runtime/memory_context.py)。
