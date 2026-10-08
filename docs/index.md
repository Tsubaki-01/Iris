# Iris 文档

Iris 是面向 Python 开发者的本地优先 Agent Kit。你用 YAML 声明模型、上下文和工具，通过 CLI 或 Python SDK 运行；工作文件、会话状态和可选的长期材料围绕本地项目组织，宿主应用负责界面与进程生命周期。

这套文档同时提供使用路线和设计路线。会 Python、初识 Agent 的读者可以从最小体验开始；已经有明确任务的读者可以直接查配方和参考。

## 开始使用

1. [运行第一个 Agent](getting-started/quickstart.md)：完成对话，再读取真实项目文件。
2. [必要概念](getting-started/concepts.md)：区分配置、宿主、会话、Run、工具与上下文。
3. [接入 Python 应用](getting-started/python-sdk.md)：复用 YAML，控制请求、结果和资源关闭。

默认不需要开启记忆、Goal、Docker 或观测。第一次运行成功后，再按需求增加能力。

## 理解设计

从[配置到一次完整运行](design/architecture.md)建立整体图景，再选择一个问题深入：

| 问题 | 设计解释 |
| --- | --- |
| 为什么仅保存聊天消息不足以恢复任务？ | [运行控制与状态持久化](design/lifecycle.md) |
| 人工回答、追加输入和取消为何不同？ | [人工交互与运行中输入](design/human-interaction.md) |
| 长对话怎样保持任务方向并回读原文？ | [上下文工程总览](design/context-engineering.md) |
| 工具从声明到执行经历哪些步骤？ | [工具生命周期与发现](design/tool-execution.md) |
| Skill、子 Agent 和命令资源怎样协作？ | [方法复用与任务委派](design/delegation.md) |
| 记忆如何进入后续任务？ | [长期记忆的组织与采用](design/memory.md) |
| 项目经验如何沉淀，如何限制修订范围？ | [经验与有限修订](design/evolution.md) |
| 目标推进和 Markdown 清单分别负责什么？ | [Goal 与 Todo](design/goals.md) |
| 更换模型协议或增加图片输入会影响哪里？ | [消息、协议与媒体](design/messages-media.md) |
| 实时输出、运行事件与 trace 有何区别？ | [流式输出与观测](design/streaming-observability.md) |
| 应选择 Hook、Middleware 还是 observer？ | [扩展点的职责](design/extensions.md) |

这些页面以概念、职责、具体流程和取舍为主。需要讲解项目时，可沿[项目导览](project-tour.md)串联产品动机、整体架构和重点专题。

## 按任务查找

| 任务 | Cookbook |
| --- | --- |
| 选择模型、拆分配置、使用 scaffold | [配置 Agent](cookbook/configure-agent.md) |
| 接入自己的 Python 能力 | [编写工具](cookbook/tools.md) |
| 连接外部工具、搜索和抓取网页 | [MCP 与外部检索](cookbook/mcp.md) |
| 运行本地命令、Python 或可选 Docker | [命令与执行环境](cookbook/commands.md) |
| 复用 Skill、配置子 Agent、处理委派结果 | [Skill 与子 Agent](cookbook/skills-subagents.md) |
| 管理多轮、steer、follow-up 与历史分支 | [会话管理](cookbook/sessions.md) |
| 处理人工问题、取消及恢复运行 | [人工交互与恢复](cookbook/hitl-recovery.md) |
| 提供固定背景、每步状态与上下文预算 | [上下文](cookbook/context.md) |
| 保存、查询并自动整理长期知识 | [长期记忆](cookbook/memory.md) |
| 生成项目经验 Skill、开放有限修订 | [项目经验](cookbook/evolution.md) |
| 持续推进一个目标、维护待办文件 | [Goal 与 Todo](cookbook/goals-todos.md) |
| 提交图片、使用语音转录 | [图片与语音](cookbook/media.md) |
| 将运行过程接入自己的界面 | [流式宿主](cookbook/streaming.md) |
| 记录和排查模型、工具与维护调用 | [观测](cookbook/observability.md) |
| 在固定事件或工具执行前后扩展行为 | [Hook 与 Middleware](cookbook/extensions.md) |

各配方写明运行前提和成功标志。确定性示例与真实模型示例分别说明；可选能力需要额外服务时，在对应页交代。

## 查询精确规则

| 参考 | 内容 |
| --- | --- |
| [配置](reference/configuration.md) | Agent YAML、进程配置、模型路由、路径和安装选项 |
| [Context 与模板](reference/context.md) | slot、动态快照、压缩、选材和项目 prompt |
| [运行与会话 SDK](reference/runtime.md) | Runner、SessionManager、SessionHistory、HITL、状态与 Store 契约 |
| [工具与扩展](reference/tools.md) | 工具开发、内置工具、MCP、命令、Skill、子 Agent、Hook、Decision |
| [记忆与长期能力](reference/memory-goals.md) | Memory、Evolution、维护协调器、Goal、Todo 与文件格式 |
| [消息与媒体](reference/media.md) | 消息类型、provider、模型事件、图片和 ASR |
| [流式与观测](reference/streaming-observability.md) | broker/gateway、wire models、SSE/WebSocket、OTel |
| [CLI](reference/cli.md) | chat 参数、交互命令和终端行为 |

## 参与贡献

从[开发入口](contributing/index.md)开始，按[源码地图](contributing/source-map.md)定位 owner。修改前查[扩展与内核边界](contributing/extending.md)，修改后使用[测试与实验指南](contributing/testing.md)选择验证范围，并同步[受影响的文档](contributing/docs.md)。

文档内容对应其所在提交的 Iris 实现；在网站阅读时，以页面标注的发布版本为准。Iris 提供运行内核和集成接口，不包含现成 Web 应用、模型部署或语音通话产品；网站展示可以复用这些 Markdown 源文件。
