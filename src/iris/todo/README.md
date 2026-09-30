# Todo：会话 Markdown 工作清单

`iris.todo` 读取每个会话自己的 Markdown 清单。文件是当前状态的唯一来源，可以由人直接
编辑，也可以用普通文件工具维护；不会新增 Todo 数据库、任务 ID 或专用写工具。

在 Agent 配置中显式启用：

```yaml
context_policy:
  enabled: true
todo:
  enabled: true
tools:
  builtin:
    - file.read
    - file.write
```

`todo.enabled` 默认 false，启用要求 context_policy 保持开启；不会自动注册文件工具或改变
写入权限。配置加载不读写 Todo 文件。

## 查询和直接编辑

```python
from iris.harness import AgentRunner

runner = AgentRunner.from_config_path("agent.yaml")
try:
    snapshot = await runner.get_todo("work")
    print(snapshot.path)
    if snapshot.error is not None:
        print(snapshot.error)
    else:
        for item in snapshot.items:
            print(item.status.value, item.content)
finally:
    await runner.aclose()
```

查询不创建 session、Run、目录或文件，不调用模型、不要求存在数据库记录。身份与 Runner
其他读取入口一样先去掉首尾空白。功能关闭抛出 `IrisTodoError`，空身份抛出 `IrisRunStateError`。
`TodoConfig`、`TodoStatus`、`TodoItem`、`TodoSnapshot` 从 `iris.todo` 导出。

当前文件位于实际 workspace 的 `.iris/todos/<session-key>.md`，session-key 是会话 ID 的
UTF-8 字节十六进制编码。例如 `work` 对应 `.iris/todos/776f726b.md`。以 snapshot.path
给出的绝对位置打开文件，不需要自己计算。相同 workspace/session 复用文件，不同 session
独立；换 workspace 不迁移文件。

```markdown
# Todo

- [x] 阅读现有实现
- [-] 完成功能
- [ ] 运行相关测试
```

`[ ]` 是 pending，`[-]` 是 in_progress，`[x]` 或 `[X]` 是 completed。`[-]` 是本清单的
进行中约定，普通 Markdown 查看器不一定渲染成原生复选框。每个项目从第一列开始、独占
一行；允许空行和 ATX 标题，不接受嵌套任务、代码块、front matter 或其他正文。

文件使用 UTF-8，可带 BOM，支持 LF/CRLF。项目顺序保留，可有重复内容和多个进行中项目。
已完成项继续保留，不再需要的项目直接移除；完成是进度声明，不是独立验收证明。

文件不存在、为空或只有标题时得到空 items；查询不会创建它。格式错误返回 error 和行号，
items 为空，不能把它解释成“全部完成”；修复后下一次查询读取新内容。其他文件读取失败
抛出 `IrisTodoError`，不会伪装成空清单。

模型编辑已有文件仍需遵守普通文件工具的先读后改规则。Todo 查询不会更新工具读记录，
不会自动修复、删除、归档或从旧聊天历史恢复文件。内部读取和解析见 [document.py](document.py)。

## 模型每步看到当前文件

每个获准的主模型步骤读取一次文件，将路径、维护说明、完整条目或格式诊断作为 required
的 `iris.todo` 动态贡献。没有宿主 context_source、文件不存在或清单为空时，模型仍能看到
当前文件位置。人或工具改动文件后，下一步骤读取最新内容。

同一步的上下文压缩和摘要重试复用已读取的快照；清单不写入原始历史、checkpoint 或摘要
原料。required 内容不能容纳时沿既有上下文预算错误处理，不截短清单。

模型在上下文中看到清单不等于调用过 read_file。普通文件工具的读取记录、权限和 stale-read
行为保持不变。文件写入成功后若工具结果提交失败，Markdown 不回滚；后续 SDK 查询读取
当前文件，工具 claim 按既有未知结果恢复路径处理，不自动重放写入。

## 同一 Run 内的一次结束自查

模型准备以无工具回复结束时，若当步清单仍有 pending/in_progress 或格式诊断，且还有模型
步骤和时间预算，runtime 会在下一步追加一次自查指令。它提醒检查遗漏、更新状态或如实
说明暂停原因，允许仍有未完成项时结束；空清单和全完成清单不触发。

真实用户 steer 优先。目标步骤从文件读取最新状态，压缩不会丢掉 required 自查指令。
checkpoint v4 只记录 `todo_reminder_step` 控制编号，不复制清单；恢复、HITL、工具步骤
继续保留编号，不安排第二次提醒。最后一个模型步骤正常接受回复，不越过预算申请自查。

流式宿主可能先收到候选回复，再收到检查后的回复；最终 RunResult 使用真正结束的回复。
Todo 不创建 Run、不启动或完成 Goal。若自查后调用文件工具，既有 Goal 报告会过时，模型
须在文件工作结束后的另一个步骤单独 report_goal；Todo 完成状态本身不代表 Goal 已完成。

## 会话、子代理和分支

同一 session 的普通 Run 和 Goal 自动 Run 沿用当前文件，每个新 Run 单独计算是否需要自查。
每个 Agent 按自己的 `todo.enabled` 启用；child 默认关闭，开启时使用自己的 session 文件，
不会继承父开关、复制父清单或自动合并结果。文件工具仍由 child 自己声明并受既有边界约束。

`SessionHistory(store).fork(source_run_id)` 产生新的 session；其 Todo 路径也随新身份改变。
新路径不存在时清单为空，存在时读取该文件。历史里保留的旧路径或工具正文不会复制成新清单。
SDK 和模型展示当前绝对路径，以该位置为准。

## 终端查看

在启用 Todo 的 `iris chat` 中输入 `/todo`，查看完成数、所有条目及实际文件路径。空文件
显示“暂无待办”，坏格式显示诊断；修改 Markdown 后再次查看即可刷新。未启用或读取失败
会显示原因，聊天继续。

`/todo` 不接受参数，不向模型发送用户输入；在执行中或等待 question/permission 时也可以
查看，之后的实际回答仍对应原交互。命令只按需读取，不监听文件或自动显示进度面板。
