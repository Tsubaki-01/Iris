# 使用 Skill 与子 Agent

Skill 复用一份任务方法，子 Agent 执行一项委派任务。前者读入当前 Agent 的上下文，后者创建独立的模型循环与会话。二者可以配合，但添加一个 Skill 不会自动产生子 Agent，配置子 Agent 也不会让它自动继承父会话历史。

## 用 Skill 保存项目方法

在 Agent 的 workspace 下创建 `.agents/skills/review-note/SKILL.md`：

```markdown
---
name: review-note
description: 将给定需求整理为问题、依据和下一步三段说明。
---

# 需求整理方法

先提取用户实际要完成的事情，再区分已经提供的事实与仍需确认的假设。
输出三个部分：问题、依据、下一步。
每个下一步都应对应前面已经解释的问题，不补充材料中没有的事实。
```

在已有 `agent.yaml` 中加入：

```yaml
skills:
  enabled: true
  root: .agents/skills
  require:
    - review-note
```

`root` 相对 workspace 解析。Iris 只扫描该目录的直接子目录；目录名使用小写 kebab-case，并包含大小写精确的 `SKILL.md`。`require` 表示这些名称必须成功发现，并不表示每次对话自动读取全文。

重新构造 Agent 后，模型上下文先出现名称与描述目录。让 Agent “使用 review-note 整理下面的需求……”时，可观察到 `load_skill` 的 `{"name": "review-note"}` 调用，返回 Markdown 正文后再生成三段说明。目录是在构造时发现的快照；`load_skill` 会读取当前文件，所以正文修改与目录新增的生效时机不同。新增、改名或修改用于目录的描述，应重建 Agent。

Skill 正文是给模型的指令材料，加载器不执行其中脚本，也不自动注册新工具。若正文要求读取 `references/` 中的文件或运行脚本，还需要提供对应文件工具或命令工具。Skill catalog 和工具 schema 各有自己的披露路径，`tool_search` 不负责搜索 Skill。

## 委派给一个专门配置的 Agent

现成示例在 [examples/subagent](../../examples/subagent/agent.yaml)。配置模型凭据后，从仓库根目录运行：

```shell
uv run iris chat examples/subagent/agent.yaml
```

输入“请委派分析：这个工具面向 Python 开发者，支持 YAML 声明 Agent”。父 Agent 默认选择 researcher，子 Agent 整理结论后返回，父 Agent 再生成最终回答。

父配置只引用目录：

```yaml
tools:
  subagent: subagents.yaml
```

目录定义默认项、selector、子配置路径和用途描述：

```yaml
default: researcher
agents:
  researcher:
    path: agents/researcher/agent.yaml
    description: 分析委派材料并返回简洁结论。
  interviewer:
    path: agents/interviewer/agent.yaml
    description: 先询问用户，再整理需求摘要。
```

目录路径相对父 YAML；目录内的 child 路径相对目录文件。每个 child 使用普通 Agent YAML，可以选择自己的模型、system、工具和 Skill。只有被选中的 child 才会加载和准备，目录描述则提前进入父模型可见的 `subagent` schema。

模型调用 `subagent` 时提供 `prompt`，可选 `agent`。省略 `agent` 使用默认项；显式 selector 必须精确匹配。prompt 要包含目标、所需材料和输出格式，因为 child 从独立 session 开始，不会自动收到父历史。父侧收到的是 child 最终 assistant 文本及结果中的 `agent_selector`、`child_run_id`；详细历史和 usage 保留在 child run，可按 ID 查询。

## 子 Agent 需要向用户提问时

在同一个示例中输入“请交给 interviewer，先确认目标用户再整理需求”。interviewer 配置了 `human.ask`，因此先产生一个问题交互。

child 等待期间，父 run 也返回 WAITING，宿主看到的是父 run 的代理 interaction，其中携带子问题。宿主依然使用父 `run_id` 与父 `interaction_id` 调用 `runner.resume(...)`，提交 `QuestionInteractionResponse`；harness 负责把它交回原 child，完成后再继续原父工具调用。不要自己启动第二个 child 来“继续”，也不要把回答作为新父 run 的普通输入。

这个行为有[跨 runner 的 SQLite 示例测试](../../tests/examples/test_subagent_examples.py)：child 提问后新建 runner，依然只通过父 interaction 恢复原 child。自己的 Web/CLI 宿主怎样渲染和提交交互，见[人工输入与恢复](hitl-recovery.md)。

## 共享什么、隔离什么

父子共享 lifecycle store，所以一次委派与唯一 child 的关联可以恢复；它们具有不同 session、run 和上下文。child 使用自己的配置装配 Hook/Middleware，不自动继承父扩展，也不开放递归 subagent 委派入口。

workspace 必须与父范围相交，最终范围取可包含的较小者；权限取父子两侧更严格的结果。命令服务由 root 拥有，child 借用，不允许重新选择 Native/Docker。共享命令环境意味着共享资源停止也可能影响其他调用，细节见[命令指南](commands.md)。

后续阅读：[Skill、委派与命令资源设计](../design/delegation.md)、[Skill 与子 Agent 参考](../reference/tools.md#skill-与子-agent)、[项目经验机制](evolution.md)。源码入口：[Skill 发现](../../src/iris/skill/discovery.py)、[委派工具契约](../../src/iris/tools/subagent.py)、[child 编排](../../src/iris/harness/_subagent.py)。
