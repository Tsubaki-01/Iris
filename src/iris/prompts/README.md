# `iris.prompts`

`iris.prompts` 将框架命名模板补齐到项目目录，并提供一次操作采用的内存快照。项目中的正文
可以手工编辑；模板目录和正文采用时机由运行装配与消费领域决定。

## 初始化与渲染

```python
from pathlib import Path

from iris.prompts import PromptSource

source = PromptSource.initialize(Path("workspace"))
snapshot = source.snapshot()
text = snapshot.render("tool_discovery_instruction", {"query_index": 0})
print(text)  # Select the best tool from state.tools for state.queries[0].
```

`PromptConfig.root` 默认 `.iris/prompts`。`PromptSource.initialize(workspace_root, root=...)`
相对 root workspace 解析该目录，也接受绝对目录；返回来源的 `root` 已固定。每次初始化仅补
缺少的种子，已有文件原样保留。临时文件写完后使用不替换目标的原子发布，两个初始化进程
竞争同一路径时不会覆盖先完成者或用户已经写入的文件。配置解析和包导入本身不初始化文件。

初始化完成后，读取只使用项目目录；删掉模板会在实际渲染时失败，不回退到包内种子。下一次
初始化会重新补齐缺失文件。目录创建、发布、读取和渲染失败均报告 `IrisTemplateError` 及路径；
消费领域负责转换为自己的异常。

## 快照与采用边界

`source.snapshot()` 捕获当前目录下全部可加载文件内容，返回 `PromptSnapshot`。快照提供：

- `render(prompt_id, variables)`：按固定 ID 渲染，ID 不含 `.j2`；变量每次调用重新传入。
- `root`：本次来源的绝对目录。
- `renderer`：冻结的 `TemplateRenderer`，可供已有 `render_file(path, context)` 调用使用。

动态 `include` / `extends` / `import`、候选 include 列表及 `ignore missing` 都从同一内存
集合取源。取快照后新增、改写或删除文件不改变当前快照。未执行的依赖分支不提前编译，坏模板
只在实际使用时失败；保持 `StrictUndefined`、纯文本默认和显式 XML 转义。快照不保存领域
变量、不缓存最终输出，也不声明多文件读取具有跨文件事务原子性。

| 消费者 | 正文采用时机 |
| --- | --- |
| Memory 自动 flush → dream → overview | 取得 Memory 锁后，整个 cycle 共用一份快照 |
| 独立 Memory 生成 SDK | 单次操作开始，调用者显式提供已初始化来源 |
| compaction 与 compaction_input | 完整压缩开始，所有批次共用一份快照 |
| memory_context、Goal、Todo、Skill 用法、Decision 指令 | runner/runtime 构造；实际领域数据仍动态传入 |
| system/context 自定义模板 | runtime 装配冻结各自来源目录；仍使用原配置声明入口 |
| project_skill_update | A 取得项目锁后，每轮固定一份模板快照，同时读取本轮策略 Skill |

child 借用 parent 的项目来源，不按缩窄 workspace 初始化另一套目录。独立
`TemplateRenderer()` 和 `ContextBuilder` 的文件更新语义保持不变；冻结能力见
[`iris.utils`](../utils/README.md)。

## 固定模板与领域职责

`PROMPT_IDS` 列出当前 14 个入口，每个 ID 对应同名 `.j2`：

| 用途 | ID |
| --- | --- |
| Memory 生成 | `memory_flush`、`memory_dream`、`memory_overview` |
| 压缩 | `compaction`、`compaction_input` |
| 动态上下文 | `memory_context`、`goal_context`、`todo_context` |
| 后续提示 | `goal_continuation`、`todo_reminder` |
| Skill 使用 | `skill_catalog_usage` |
| Decision | `memory_recall_instruction`、`tool_discovery_instruction` |
| 项目经验整理 | `project_skill_update` |

默认种子与 Python 包一起分发。`prompts` 仅负责来源与渲染入口，不导入消费领域；变量、请求
角色、结构化输出 schema、解析及应用规则由各领域持有。Memory 的固定输出要求和实际模型
schema 直接加入请求，不依赖可编辑模板保留相关文案。项目经验 A 阶段同样由 evolution
追加正文/no-change 的固定协议与响应 schema。

当前支持手工修改项目正文；自动修订不属于此包已实现能力。初始化与并发发布测试位于
`tests/prompts/`，完整 Jinja 快照语义测试位于 `tests/utils/test_templating.py`。
