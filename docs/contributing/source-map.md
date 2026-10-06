# 按改动目的查找源码

这里按职责定位代码；完整理解一次运行先读[架构总览](../design/architecture.md)。各包 README 是进一步阅读材料，具体接口与默认值仍以当前实现和[参考手册](../index.md#查询精确规则)为准。

## 声明与模型边界

| 包 / 文件 | 主要职责 | 常见修改 |
| --- | --- | --- |
| [agents](../../src/iris/agents) | Agent YAML 模型、解析及声明式注册 | 配置字段、引用解析、装配输入 |
| [config.py](../../src/iris/config.py) | 进程配置、凭据与 provider 注册 | 全局配置字段与读取入口 |
| [context](../../src/iris/context) | 结构化 context 与动态 source 契约 | slot、模板渲染、宿主快照 |
| [message](../../src/iris/message) | provider-neutral 消息、请求、响应和模型流事件 | 新内容类型、标准响应语义 |
| [providers](../../src/iris/providers) | 协议投影、服务调用与响应解析 | Responses/Chat 适配、错误和 usage |
| [prompts](../../src/iris/prompts) | 项目命名模板初始化与操作快照 | 默认策略模板、模板采用边界 |
| [templates](../../src/iris/templates) | 内置 scaffold | 新项目文件模板与生成行为 |

## 完整运行与持久状态

| 包 | 主要职责 | 常见修改 |
| --- | --- | --- |
| [harness](../../src/iris/harness) | AgentRunner、SessionManager、SessionHistory、维护协调 | start/resume/recover/cancel、输入准入、分支与共享服务 |
| [runtime](../../src/iris/runtime) | 一次 activation 的推进、上下文选材、工具桥接 | 模型步骤、检查点游标、执行阶段与提交端口 |
| [lifecycle](../../src/iris/lifecycle) | 不可变模型、状态转换命令、存储协议 | Run/Session 契约、修订号、查询投影 |
| [store](../../src/iris/store) | 内存与 SQLite 的统一协议实现 | 持久化、查询、恢复解析 |
| [hitl](../../src/iris/hitl) | 交互与回答类型、无状态领域服务 | permission/question、交互身份与回答语义 |

Runtime 不选择数据库；store 不调度模型；host 不维护另一份权威 checkpoint。跨这几层修改时，先确定新增信息是请求、当前状态、提交命令还是结果投影，避免同一事实出现两个 owner。

## 能力执行与长期工作

| 包 | 主要职责 |
| --- | --- |
| [tools](../../src/iris/tools) | 工具 schema、registry、执行、权限、discovery、artifact 和内置工具 |
| [mcp](../../src/iris/mcp) | 外部 MCP 配置、连接与工具适配 |
| [command](../../src/iris/command) | Native/Docker 命令服务、期限与停止协调 |
| [sandbox](../../src/iris/sandbox) | Docker 引擎、容器与挂载的物理生命周期 |
| [skill](../../src/iris/skill) | 项目 Skill 发现、目录和按需加载 |
| [hooks](../../src/iris/hooks) | 固定事件扩展声明与派发 |
| [decision](../../src/iris/decision) | 可选 Choice/Boolean/Score SDK 与接点配置 |
| [memory](../../src/iris/memory) | 长期知识、概览、Search/Fetch 与经历生成流程 |
| [evolution](../../src/iris/evolution) | 项目经验 Skill 与显式开放的 prompt/config 修订 |
| [goal](../../src/iris/goal) | 持久目标与跨 Run 的推进规则 |
| [todo](../../src/iris/todo) | 会话 Markdown 清单的读取和上下文投影 |

子 Agent 的委派声明与工具形状在工具相关模块，child runner 的运行控制在 harness。不要因为它“是一个工具”，就让 ToolExecutor 接管子任务的完整生命周期。

## 宿主适配与支撑

| 包 | 主要职责 |
| --- | --- |
| [cli](../../src/iris/cli) | 终端输入输出、slash commands、typed HITL 和进程收尾 |
| [streaming](../../src/iris/streaming) | live broker、会话 gateway、SSE/WebSocket framing |
| [observability](../../src/iris/observability) | 标准 OTel 记录与导出，不接管执行状态 |
| [speech](../../src/iris/speech) | ASR adapter 与 PCM 转录入口 |
| [exceptions](../../src/iris/exceptions) | 领域异常 |
| [utils](../../src/iris/utils) | 模板、图片、文件及后台执行等跨领域基础工具 |

## 用会话分支追一次真实修改链

想理解“从某次回答开始另一条对话”，可以依次阅读：

1. [SessionHistory](../../src/iris/harness/session_history.py)：公开的历史查询、截点和 fork。
2. [历史模型](../../src/iris/lifecycle/history.py)：截点、分页和返回视图。
3. [LifecycleStore](../../src/iris/lifecycle/store.py)：`ForkSession` 等命令与存储契约。
4. [store 实现](../../src/iris/store)：两种后端如何执行相同语义。
5. [SessionHistory 测试](../../tests/harness/test_session_history.py)：哪些历史被复制，哪些执行状态不能继承。

若新需求只是改变界面的分支按钮，不必修改 store。若改变截点语义，则需要一起调整公共模型、两种存储和相关测试；不能只在 CLI 中拼一份看似相同的消息列表。

下一步：[选择扩展点](extending.md) · [精准验证](testing.md)。
