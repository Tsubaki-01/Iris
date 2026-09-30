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
