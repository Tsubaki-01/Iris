[English](README.en.md)

# `iris.harness`

`iris.harness.AgentRunner` 是 Iris 唯一的 complete-run SDK facade。它拥有 logical run 的创建、
resume、durable cancellation、settlement observation、显式 recovery、事件投递和 activation
live resources；`AgentRuntime` 只作为其内部 engine。`SessionManager` 是可选的单 session
process-local admission facade，只组合 runner，不接管 durable ownership。

## 快速入门

```python
from iris.harness import AgentRunRequest, AgentRunner

runner = AgentRunner.from_config_path("agent.yaml")
try:
    result = await runner.start(
        AgentRunRequest(input="你好", session_id="default")
    )
    print(result.run.phase, result.assistant_message)
finally:
    await runner.aclose()
```

`from_config*()` 使用配置文件目录解析相对路径。显式传入 `store=` 时，runner 的所有 durable
reads/writes 使用该 exact object；否则 `session.backend: none` 选择
`InMemoryLifecycleStore`，`sqlite` 选择 lifecycle `SQLiteStore`。

两个 `from_config*()` 入口都接收可选 `decision_client=`，只借用其 `evaluate` 能力，关闭 runner
不会关闭注入对象。配置自建的 Jev 客户端跨 session/Run 复用并随环境关闭；子 Agent 根据自身
配置独立构造，不继承 root 的注入。接点开关和配置示例见 [Decision](../decision/README.md)。

在尚未关闭的 runner 上，图片输入先导入，再提交完整数据块；主模型需支持所选协议的视觉输入：

```python
from pathlib import Path
from iris.message import TextBlock

image = await runner.import_image(Path("invoice.png"), session_id="default", name="发票")
result = await runner.start(
    AgentRunRequest(input=[TextBlock(text="这张发票的金额是多少？"), image], session_id="default")
)
```

`import_image(Path | bytes, *, session_id, name=None)` 将相对路径按 runner workspace 解析，
在线程中完成图片处理并保存到 `.iris/image-cache/<session 编码>/`，返回 `ImageBlock`。
导入不创建 run 或占用 session lane；来源文件后续改动不影响副本。纯图片 `[image]` 同样有效，
空字符串、空列表和只有空白文字的列表不接受。`SessionManager.submit()` 的 idle、steer、
follow-up 接受同一 `str | list[DataBlock]` 输入；先在锁外完成导入，入队只传已保存引用。
start、HITL resume 和 recover 均使用 durable request/历史中的完整块，无需再次导入。
关闭 runner 不清理图片缓存；SQLite 重启恢复与 fork 继续依赖这些文件，备份时须同时保留
`.iris/image-cache/`，仅复制数据库不足以恢复图片，也不会自动改写跨机器路径。
`ContextBuildScope.run_input` 保持纯文字，纯图片的 `ForkPoint.input` 显示 `[image: 名称]`；
Goal continuation 和 subagent prompt 仍为文字，不自动携带父图片。

配置 MCP 时构造不连接；`aprepare()` 可显式预热，也会由首次执行入口自动调用。完整目录发布
后才创建 run；required 准备失败会关闭资源且不创建 run，修正后需新建 runner。

同一 runner 的并发入口共享一次资源准备；取消某个等待者不取消共享准备。关闭 runner 会先收口
已有准备任务，再关闭自有资源。每次实际 Run 的 memory 前台登记仍独立执行。

启用 `goal.enabled` 时，Runner 用同一 lifecycle store 装配 GoalService、目标工具与动态上下文。
Goal 单轮入口复用普通 start 的执行注册、steering 和命令生命周期；自动输入标记为 context，
不会替换最近普通用户输入锚点。`start()` 仍只执行一次 logical run。目标模型与申报规则见
[Goal 说明](../goal/README.md)。

启用后，`SessionManager(runner, session_id).goal` 提供异步 create/get/edit/pause/resume/
complete/clear。创建显式允许续跑；后继不进入用户 FIFO，已接纳的用户工作先执行。暂停只停止
后续轮次，`interrupt()` 同时暂停目标并按原契约取消当前 Run；仅停止空闲 Goal 时返回 None。
关闭功能时 `manager.goal` 为 None。

完整配置与 `Runner → Manager → create/get/pause/resume` 示例见
[Goal SDK](../goal/README.md#从配置到执行)。`GoalSession`、`GoalView`、`GoalControlResult`、
`GoalChanged` 可直接从 iris.harness 导入。控制回执分别表示 scheduled/admitted/running/
waiting/needs_recovery/occupied/stopped，scheduled 不代表已经执行。轮数在准入消耗且不退款，
resume 不重置；Run deadline_at 仍是绝对时刻。完成来自主模型或用户申报，不是独立认证。

新 manager 默认不续跑持久 active 目标。显式 `goal.resume()` 可以附着原 WAITING Run、恢复
其 deadline timer 并回答原交互；无人推进的 ACTIVE Run 要提供 `expected_activation_id`。
恢复不增加轮数。后台 deadline 和命令清理错误同样触发 Goal 状态通知，不依赖 live publisher。
命令清理尚未完成时仍占原 lane，须显式重试原收尾。
WAITING 的恢复沿用 `manager.resume(interaction_id=..., response=...)` 回答原 typed 交互，
不会创建新 Run。默认 child 和 history fork 不继承目标，child 显式启用 Goal 会被拒绝。

`GoalChanged` 是最新状态快照：mixed 流先交付相应 terminal，再交付目标结果；broker-only
发布 session-scope `goal.changed`。读取状态不会补结算或启动工作。mixed tracker 满时停止准入，
消费释放后再推进。Goal 与用户 follow-up 共用现有 memory handoff；临时预留只覆盖交接，
等待用户/HITL/容量不阻止原闲置整理。manager 关闭注销控制附着，不接管 Runner 的资源关闭。

resume/recover 先保留纯 durable 结算，确需执行才准备；准备后重新读取状态、checkpoint、
claim 与时间，使用当前 runner 的配置继续。terminal 读取、普通 waiting 到期、未结算 CLAIMED 的 unknown
恢复、查询、history fork 和取消申请不依赖 MCP 连接。

创建 run 和 resume/recover 的 checkpoint 检查使用 `load_session_revision()`，不为取得版本号
加载完整历史；检查仍与 store 当前 revision 独立比较。Commit port 初始化复用已经读取的
checkpoint revision。输入准备通过绑定 port 的 `load_session_header()` 只取元信息；主模型步骤
通过 `load_model_context(include_tool_discovery=...)` 读取一致的有效历史快照，两个入口都更新
port 持有的 session revision。后者由 store 的 `load_run_context(run_id, ...)` 提供摘要、未覆盖
后缀及当前 run 保护锚点，按需包含发现投影。公开 `get_session()` 仍返回完整原文。

root 连接跨 run 复用。host 停止新调用后，须等待原 start/resume/recover 完整返回再 `aclose()`；
cancel 的 terminal 已包含必要命令收尾，但不代表原调用的事件投递已经退出；观察超时也不代表结算。
active 时关闭会报错，
关闭成功后重复关闭幂等；清理失败保留 pending 和资源关闭状态，再次 `aclose()` 会继续清理。
开始关闭后不再接收业务调用，但仍可查询 durable 结果。`SessionManager.close()` 不接管 runner 资源，
使用它时先 `close(cancel_run=True)` 再关闭 runner。

child 的普通 YAML 可独立配置 MCP。fresh child 在 admission 前准备；准备失败返回
`SUBAGENT_PREPARE_ERROR`，不创建 child run/link。每次 child WAITING/结束、恢复失败或提前返回
均关闭本次 child 自有资源，下次 resume/recover 按当前配置重建。root 命令服务、绝对期限
timer 和失败清理 pending 不随临时 child runner 关闭；它们通过 exact child run 与 route 重建结算。
父子连接独立，父子权限仍取更严格的组合。live cancel 借用当前 runner 并等待原任务，资源由
创建它的作用域关闭；关闭异常记日志并保留原结果。非 live 取消先写 durable request，再按需准备。

## Run Hooks

`AgentRunner.from_config()` 与 `from_config_path()` 都接收 `hooks=` 与 `tool_middlewares=`：
前者是 `HookRegistration` 序列，后者是已构造的 `ToolMiddleware` 实例序列，默认均为空。
配置 `hooks` / `middleware.tools` 在前、SDK 项追加在后，不按名称覆盖或去重。
同次装配只创建一份实例；父 SDK 项不自动传给 child。完整配置与命令协议见
[Hooks](../hooks/README.md)，工具包装契约见 [Tools](../tools/README.md)。

```python
from iris.hooks import HookEvent, HookRegistration


async def report_finished(event: HookEvent) -> None:
    print(event.event, event.run_id)


runner = AgentRunner.from_config_path(
    "agent.yaml",
    hooks=[HookRegistration(
        event="run.finished", name="report-finished", handler=report_finished
    )],
)
```

环境中的同一 `HookDispatcher` 同时供工具执行器与 harness 使用。`run.started` 在资源准备、
Run 准入和 active task 注册后、首模型调用前运行；普通、Goal 和 child 开始共用这个位置。
准备失败或 Goal 未获准时不派发，WAITING、resume 和 recover 不重发 started。
开始处理器处于原取消与绝对期限管理内；真实命令 unknown、清理失败和 SDK task 取消保留各自
原有语义，不制造工具 claim 或伪造 engine checkpoint。

`run.finished` 只由本次新终态提交触发，包括 outcome-ready recovery 的直接终结。
Python 处理器适用所有终态，命令处理器只适用 COMPLETED；普通失败只记录，原 durable result
不变。原命令结算完成、pending 移除后才运行 finished；它使用自身期限，不再受已结束 Run 的
deadline 限制，也不等待慢 observer。历史结果读取不补发；它不是资源释放或必达保证。

存在适用的 finished 处理器时，root 在终态可见前登记完成任务。同 session 的直接 SDK 新 Run
会暂时报 `IrisRunStateError`；`SessionManager.submit(mode=None/auto)` 在锁外等待，醒来后
用原参数重新准入，不预留名额。显式 steer 不能发送给 terminal Run，follow-up 保持 FIFO，
完成通知主动唤醒队列。Goal 保留原 managed-task 等待，并观察共享完成状态与准入错误。
没有适用 finished 处理器时，保留原 observer 与后继 Run 的并发行为。

取消普通 submit 等待者或默认 Manager detach 不取消结束处理；取消实际 start/resume/recover
驱动或 `close(cancel_run=True)` 则中断 finished 并排空当前命令后返回。后终态清理失败会报告
`IrisCommandCleanupError` 并拒绝 root 下的新 Run；已有 cancel/resume/recover 与资源关闭仍可
使用。它不进入可重试 pending、不再提交终态，宿主应等已有驱动退出并关闭 root 后重建。

live 与 rebuilt child 借用 root 的完成 owner，各自使用本 Agent 的处理器。child 在准备完成、
同步准入前也检查 root 错误；临时 child 关闭不销毁共享任务。root `aclose()` 会等待这些实际
任务，包括原 child 已关闭后由后台期限触发的 finished，然后关闭共享命令服务。

## 会话 Todo 读取

启用 `todo.enabled` 后，`await runner.get_todo(session_id)` 返回当前工作区的 Markdown
清单，包含绝对 `path`、冻结 `items` 和可选格式诊断 `error`。接口使用实际 runtime 的配置
和工作区，显式 runtime 构造也适用；不创建 session/Run，不占执行 lane，不准备 provider，
不改变历史版本。人工修改后下次查询即可看到新内容。文件不存在时为空，读取失败与未启用
使用 `IrisTodoError`。格式及示例见 [Todo](../todo/README.md)。

同 session 的普通 Run 和 Goal 自动 Run 复用当前文件，每个 Run 分别拥有一次自查机会。
子代理按自己的配置独立启用，用其原 child session 定位清单；恢复不换文件身份，也不继承
或合并父清单。`SessionHistory(store).fork(source_run_id)` 返回新 session，其新 Todo 路径
不存在时为空，已存在则读取实际内容，不复制原历史中的清单。
直接使用 `RuntimeFactory` 仍需提供 context_policy 原有的 `context_access`，Todo 不新增
factory 参数或服务。终端通过 `/todo` 按需调用同一 SDK 查询，等待 HITL 时也不消费人工回答。

## 当前会话上下文回读

`context_policy.enabled` 默认 `true`。`AgentRunner.from_config*()` 使用同一个 lifecycle store
构造内部 `ContextAccess`，由共同 runtime 装配入口注册 `context_read` 与 `context_search`。
宿主不必配置 `file.read`、memory 或额外存储；设置 `enabled: false` 可关闭这两个工具。

回读使用本次工具执行的 session ID；root 与 child 各自读取自己的已提交历史，不自动读取
parent。`message:<index>` 和 `result:<message_index>:<block_index>` 都使用从零开始的原始
位置，压缩后的摘要不改变编号。Fork 继承前缀位置和原 artifact 引用，文件不复制。

小结果从 lifecycle 原文读取，大结果通过已提交 artifact 取回最终文本或原生 MCP JSON；
内联文字与图片引用按原顺序组成分页文字，图片引用包含名称、MIME、original/model 路径和尺寸。
Search 可以匹配图片名称与引用；它只查已提交正文和预览，不读取像素或扫描完整 artifact 文件。
读取不会重跑来源工具；已有 text_path 直接分页该文件，不在每页重复追加图片引用。具体分页参数、
错误与范围见 [tools 说明](../tools/README.md#当前会话上下文回读)。读取服务位于 harness，
runtime 只接收窄接口，不取得 store ownership。

图片随消息参加普通主请求；看过一次不会自动移出。压缩保留当前 run 初始输入、最新 steer
和保留尾部中的完整图片。摘要调用只收到已有文字和图片引用，摘要模型不必具备视觉能力，
也不能根据引用补出历史尚未表达的图片细节。旧前缀图片随其消息退出活动上下文后，可以先
`context_read` 获取 model 路径，再调用已配置 `file.read` 的 `read_file` 重新带回图片。
未配置文件工具时仍可读取引用，由 host 重新提交已有 `ImageBlock`；框架不自动挂载文件工具。

## 宿主动态上下文

`AgentRunner.from_config()` 与 `from_config_path()` 接收可选 `context_source=`。它实现
`iris.context.ContextSource.collect(scope)`，由 runtime 在每个获准的主模型步骤采集一次当前
状态；宿主负责内容、`required` 和 `priority`，示例见 [context 说明](../context/README.md#宿主动态快照)。
`context_policy.enabled=false` 与 source 同时提供会在装配时报 `IrisConfigError`。

source 绑定当前 runner，同 runner 的多个 session 可并发采集，因此应用应按 scope 中的
session/run 区分自己的状态。各步骤快照只进入本次请求，不写入 lifecycle history/checkpoint；
它补充当前应用状态，BCI 仍保留本次 run 发起时的背景。恢复到 `before_model` 会重新采集，
child 不继承 parent 的 source 或快照。需要持久证据时使用普通工具结果或宿主文件。

## 按需工具披露与恢复

`context_policy.deferred_tools: true` 自动接入 `tool_search`。候选在成功搜索结果提交后才供
下一主模型请求选择完整 schema，披露事实保存在本 session 原始历史；多个 session 共用
registry 不共享已披露集合。Fork 只继承复制前缀内的发现，child 不继承 parent 的披露状态。
MCP 仍在执行前完整 prepare，按需 schema 不推迟连接或发现。

checkpoint v4 的 `engine_cursor.visible_tool_names` 保存产生当前工具批次的可见名称，
与 assistant calls 一起提交。WAITING、部分工具完成和恢复均保留该集合，不重跑 source
或 schema 选材；工具权限仍在执行前刷新。批次完成后清空，下一个主步骤重新选择。
预算、首项保护及 forced tool 规则见 [runtime 说明](../runtime/README.md#按需工具定义)。

## 会话历史分支

`SessionHistory(store)` 复用与 runner 相同的 `LifecycleStore`，提供三个同步方法。它只生成新
session ID 和创建时间，查询、来源资格与原子复制由 store 负责，领域异常原样传给 host。
它不接管 store 的关闭或 runner 的资源生命周期。

| 方法 | 返回值与语义 |
| --- | --- |
| `list_fork_points(session_id, *, after=None, limit=50)` | `ForkPointPage`，按 `(created_at, run_id)` 升序；`after` 接受上一页的 `ForkPointCursor`，`limit > 0` |
| `get_at_run(source_run_id)` | `RunHistorySnapshot`，包含分支点及其末尾的已提交历史，不提供当前 session CAS revision |
| `fork(source_run_id)` | `SessionSnapshot`，自动生成目标 ID 并创建新 session；每次成功调用都产生不同分支 |

上述历史 DTO 从 `iris.lifecycle` 导入。列表没有合格 run 时返回空页，没有后续页时
`next_cursor=None`。以下代码在 host 的现有 async 函数中执行，前提是 `main` 已有至少一个
terminal 顶层 run：

```python
from iris.harness import AgentRunner, SessionHistory
from iris.lifecycle import AgentRunRequest
from iris.store import SQLiteStore

store = SQLiteStore(".iris/session.db")
runner = AgentRunner.from_config_path("agent.yaml", store=store)
history = SessionHistory(store)

page = history.list_fork_points("main", limit=20)
if page.next_cursor is not None:
    next_page = history.list_fork_points("main", after=page.next_cursor, limit=20)

point = page.items[0]
preview = history.get_at_run(point.run_id)
branch = history.fork(point.run_id)
result = await runner.start(
    AgentRunRequest(input="Try another approach.", session_id=branch.session_id)
)
```

来源接受全部 terminal stop reason；linked child 被拒绝，调用过 child 的顶层 parent 仍可
使用。复制只到所选 run 的终态消息截点；来源 session 忙于后续 run 时，也可从旧截点创建分支。
返回的新 session 从 `revision=0` 开始，`forked_from_run_id` 保存直接来源，后续追加保留来源。
目标 ID 使用 `session_` 前缀与 UUID，`fork()` 不接收目标 ID 参数。

Fork 本身不调用 provider 或创建 run，复制终态原文截点、当时冻结的摘要投影和直接来源。
来源 session 后续压缩不改变旧 run 的分支摘要。下一次 `start()` 使用 host
选定 runner 的当前 system、工具、Skill、memory 与 workspace 配置，不恢复旧 checkpoint 或
复制旧运行环境。消息中的 artifact 引用保持原值，文件不会随 fork 复制。Host 也可直接构造
`SessionManager(runner, branch.session_id)` 并调用 `await manager.submit("Try another approach.")`，
按既有事件流观察运行并在结束使用时关闭 manager；无需 attach 或切换原 manager。

实现位于 `session_history.py`，相关集成用例位于
`tests/harness/test_session_history.py`。

## 公共操作

`iris.harness.ChildProviderFactory` 定义 selected child provider 注入协议：
`__call__(config: AgentConfig, *, config_path: Path) -> CompletionProvider`。
`CompletionProvider` 从 `iris.providers` 导入，要求 `complete()` 与 `estimate_input_tokens()`。
该工厂接收已加载的普通 child 配置，不依赖 parent provider 的单次凭据覆盖。

`from_config*()` 接受 `permission_policy=` 与 `child_provider_factory=`。配置 catalog 时，
runner 读取一次路由快照并装配内部 controller。Selected child 使用普通 AgentConfig、独立
session/run、fresh `AgentRunOptions()`、空 request metadata，并共享 parent store/clock。
Child 按自己的 memory 配置和 effective workspace 构造 service，不继承父 service 或动态快照。
Child 不注册 subagent；linked ACTIVE 通过 ordinary recover 继续原 child，
WAITING/TERMINAL 只读原结果。Child 等待时创建 parent proxy，工具保持 PREPARED。Host 只向
parent `resume()` 提交回答；回答先持久化，再继续 exact child。再次等待只替换 proxy，最终结果
通过单次 finalize 推进 parent cursor，使用 parent identity/artifact 与 tool error policy。

回答已持久化但推进中断时，`recover(parent_run_id)` 从 RESOLVED proxy 或 outer permission
恢复；普通 PENDING waiting 仍需 `resume()`。ACTIVE recovery 仍要求 activation fence。
新进程按 durable selector 取当前 catalog 快照，继续原有 parent/child run。
Linked continuation 不重复 outer permission；未 admission 的存储批准仍执行 permission refresh。
WAITING finalize 成功后先发布原工具 activation 的 `tool.completed`，再执行 fresh RESUME
activation。SessionManager 在第一次 child await 前完成 resume admission，并拒绝并行回答。

Parent 取消、deadline 或 parent-owned proxy 到期先结算 exact child，再结束 parent。
Linked proxy 的 `request_cancel()` 返回仍为 WAITING 的请求快照；`cancel()` 或 manager
settlement task 完成后才 terminal。`settlement_timeout` 覆盖 child 等待和 parent observation，
超时保留 durable cancellation，后续 `cancel()` / `recover()` 可继续结算。child 的停止收据只交给
当前父调用；即使 child 已由 timer 终态，父 proxy 延后处理仍消费原收据，不重停后续环境。连续 interrupt
共享原 cleanup task，follow-up 等到 parent 真正 terminal 才启动。

Child interaction/deadline 或 outer tool timeout 先结算 child，再提交 `SUBAGENT_TIMEOUT`。
Outer timeout 以 durable `child.created_at` 为起点，权限等待不计入，resume/rebind/recover
不重置。Proxy 保存的 expiry owner 决定停止 parent 还是继续处理工具错误；parent 同刻到期优先。
Parent stream 只含 parent start/proxy/final facts；linked recovery 不重复 started，child 内部
stream 与 usage 不转发，错误仍用带 `is_error` 的 `tool.completed`。
Parent 保留最终回答权，只得到 child 的最终文本；child usage 留在 child run。
结果 metadata 仅含 `agent_selector`、admission 后的 `child_run_id` 和普通 artifact metadata。
Live event 是 best-effort，不承诺跨进程 exactly-once。

最小 Sub Agent 示例包含三个文件，使用前述 `AgentRunner.from_config_path("agent.yaml")`
运行 parent；模型凭据按既有 provider 配置提供。

```yaml
# agent.yaml
name: parent
model: openai/gpt-4o-mini
system: 将专注的子任务交给 researcher，再结合其结果给出最终回答。
tools:
  subagent: subagents.yaml
```

```yaml
# subagents.yaml
default: researcher
agents:
  researcher:
    path: agents/researcher/agent.yaml
    description: 梳理专注的子任务并给出简洁结论。
```

```yaml
# agents/researcher/agent.yaml
name: researcher
model: openai/gpt-4o-mini
system: 只处理交给你的子任务，必要时向用户提问，最后返回结论。
tools:
  builtin: [human.ask]
```

Host 收到 parent `pending_interaction` 后，仍按普通 typed HITL 调用 parent `resume()`。
无需操作 child runner。恢复按已保存的 selector 查找当前 catalog 中的 child 配置；
catalog 描述变化不阻止恢复，也不会新建替代 child。
Child 已关闭 HITL interaction 但尚未提交工具结果时，普通 ACTIVE recovery 仍从该 interaction
恢复存储回答。Child 的 `IrisRunRecoveryError` 原样传播，保留 parent/child 的可恢复状态。

- `start(request, options=None)`：原子创建 run/start activation，并推进到 waiting 或 terminal；
- `resume(run_id, interaction_id=..., response=...)`：消费 exact waiting interaction；
- `request_cancel(run_id, reason=None)`：只保证首次请求持久化；active 本地 activation 在提交后
  才收到 signal，waiting 保留原状态，由异步 `cancel()` 清理后 terminal；
- `cancel(..., settlement_timeout=None)`：request + 观察 durable terminal result；观察超时不写
  新事实；观察到结算不代表原 `start()` / `resume()` 调用已经退出；
- `recover(run_id, expected_activation_id=...)`：对 active run 要求精确 fence。safe checkpoint
  创建 recover activation，outcome-ready 只补 terminal，unresolved claim 结算为
  `outcome_unknown`；
- `get_session()`、`get_run()`、`get_run_control()`、`get_result()`、`list_tool_calls()` 和
  `list_events(after_sequence=0, limit=None)`：无副作用 durable reads；`limit` 如提供必须是
  正整数。

waiting run 应使用 `resume()`，不是 `recover()`。terminal run 的 cancel/recover 是幂等读取。
`resume()` 将当前 interaction ID、run revision、interaction version 与 typed response 交给 Store；
不再回传 interaction 内已有的调用指纹作为 resolve 参数，实际工具执行的指纹绑定仍保留。

`get_run_control()` 只读取 run identity、phase、activation fence、revision 与取消控制字段。
SessionManager 在锁内用它判断 steer 是否仍可进入当前 activation，不装载完整 run snapshot。
事件补读直接消费 store 返回的有序、唯一、有限页；store 是排序和分页的唯一 owner，manager
只推进 watermark 并保留空页冲突与 submission barrier。

内部 `RunCommit.session_revision` 只返回本次会话 revision；commit port 直接更新本地 revision，
mutation 不为回执加载完整 history。需要消息时仍显式调用 `get_session()`。

## Live publisher 组合

Host 可把同一个 `LivePublisher`（通常是 `LiveStreamBroker`）通过 `live_publisher=` 注入
`AgentRunner`。Runner 会把每个 activation 的 `RuntimeStreamEvent` 和每条新 committed
`RunEvent` 同步交给 publisher；未注入时不构造 runtime sink。Publisher 是 best-effort
观察面：普通异常只记录不含 payload 的 warning，不会回滚 durable mutation、取消 run 或改变
`RunResult`。

是否注入 publisher 是 runtime transport 的唯一选择：未注入时 complete，注入时 streaming。
`ModelConfig` 不包含 `stream`；低层直接调用 provider 时仍由 `LLMRequest.stream` 指定接口。

`AgentRunner.from_config()` 与 `from_config_path()` 只把 publisher 交给 runner，
`RuntimeFactory` 不拥有 broker 或 fan-out。Host 可通过 `get_session()`、`get_run()`、
`get_result()`、`list_tool_calls()` 和 `list_events()` 从 exact runner/store 补读 durable facts，
这些读取不会触发 live 发布。

## 单 session 输入管理

`SessionManager(runner, session_id, submission_publisher=...)` 绑定一个 exact runner 与一个
session。它适合需要在当前 run 执行期间接收新普通输入的 host：

Host 显式选择 `observation_mode="mixed"`（默认）或 `"broker_only"`。Mixed 模式提供
lossless `events()`，也可同时发布到 broker。Broker-only 模式要求 `submission_publisher`，
只发布 submission facts，不创建本地 tracker/transient buffer；调用 `events()` 会失败。
仅使用 Gateway 的 host 应选择 broker-only，容量不会取决于一个未使用的本地 consumer。

```python
import asyncio

from iris.harness import AgentRunner, SessionManager, SubmissionEvent

runner = AgentRunner.from_config_path("agent.yaml")
manager = SessionManager(runner, "default")

async def consume_events():
    async for event in manager.events():
        if isinstance(event, SubmissionEvent):
            print(event.submission_id, event.state, event.reason)

consumer = asyncio.create_task(consume_events())
initial = await manager.submit("先分析现状")
queued = await manager.submit("把重点改为并发边界", mode="steer")

# host 结束使用 manager 时：
await manager.close()
await consumer
```

Idle 时，`submit(input, mode=None, options=...)` 在 run create 已 durable commit 后返回
`SubmitReceipt(state="delivered")`，但不等待 provider 或 run settlement。Busy 时必须显式选择：

- `mode="steer"`：绑定 exact current run，不接受新 run options；runtime 只在安全边界 claim
  一条，成功写入 durable session history 后才产生 `SubmissionEvent(state="delivered")`；
- `mode="follow_up"`：预生成 future run id，可携带 options；只在 exact current run terminal 后
  串行创建，一次启动一条。

CLI 等普通文本 host 可用 `mode="auto"`：manager 持锁读取当前 durable 状态，idle 时新建 run，
busy 时选择 steer。此时 `options` 只用于新 run，busy 分支不使用它；UI 无需维护路由状态副本。

两种 mode 各自保持 FIFO，但按 eligibility 独立推进，因此较早的 follow-up 不阻塞仍可进入当前
run 的 steer。Busy receipt 只表示 `pending`；最终 delivery/failure 通过所选 observation 模式报告。
该单消费者 stream 原样混合 durable `RunEvent` 与 transient `SubmissionEvent`，不创建 session-global
sequence。Idle submit 不产生 `SubmissionEvent`。

Mixed 模式下，可选 `submission_publisher` 提供 submission side channel。Manager 会先把原
`SubmissionEvent` 成功写入上述单消费者 buffer，再 best-effort 发布带 session identity 的
`SessionSubmissionEvent`；发布失败不重复 buffer 写入，也不改变 receipt。Run events、HITL 和
result 仍以原 manager/runner 契约为准。

Manager 默认最多分别排队 64 条 steer 与 64 条 follow-up，最多保留 256 个 transient submission
event 槽位，并跟踪 64 个尚未被 consumer 追平的 durable run；后两种限制仅用于 mixed 模式。可通过
`max_pending_steer`、`max_pending_follow_up`、`max_buffered_submission_events` 和
`max_tracked_durable_runs` 关键字参数设置其它有限正整数。Busy admission 会同时预留 pending 与
terminal event 槽位；任一容量不足时，在 receipt、队列和 event 发布前抛出 `IrisRunStateError`，不
静默丢弃。已接纳的 follow-up 在 tracker 暂满时保留 FIFO，consumer 追平旧 run 后继续推进；新的
idle submit 则在创建 task 前拒绝。Durable tracker 的容量判断与 baseline 登记由同一个同步
admission mutation 完成，不使用 check-then-register 双阶段路径。

HITL response 通过 `manager.resume(interaction_id=..., response=...)` 等待完整结果，或通过
`admit_resume(...)` 在 activation 已接纳后返回 `ResumeReceipt(run_id, interaction_id)`。
两者共享同一 admission owner；后台执行仍由 manager/runner 持有，不进入普通输入队列。
`interrupt()` 先暂停 Goal 续跑，再请求取消 exact current run；无 Run 但暂停了 Goal 时返回 None。
active cancellation request 不是 terminal，follow-up
仍等待真实 settlement。WAITING 或清理失败后已无活动 continuation 的 ACTIVE run，由 manager
持有唯一异步 cancel task；再次 interrupt 可重试 pending 清理。新 run 不继承旧 run 的 cancel owner。
`close()` 拒绝后续操作、以 `session_closed` 结算全部 pending input 并结束
event stream，但不取消或等待当前 run。

即将关闭 event loop 的 host 使用 `close(cancel_run=True, reason=...)`：先关闭 admission 并
失败掉 pending input，阻止启动下一条 follow-up，再通过 runner 取消并等待当前 run 结算。
随后等待原 managed task 结束，包括 WAITING parent 正在继续等待 child、旧 terminal 尚在投递事件
的情况。清理失败后 admission 保持关闭，但原 run/task 引用仍保留；再次 `close()` 只重试收尾，
成功后重复关闭才幂等返回。取消某个 close 等待者不会取消 manager 持有的关闭任务。
即使关闭清理失败，mixed event stream 也会结束，避免 host 的 consumer 阻挡退出。
CLI 的 `/exit`、EOF、Ctrl-C 和错误退出均使用这条路径。

Queue、receipt 状态、submission events、claim 和 durable event 水位都只存在于当前进程。Durable
event payload 不进入无界进程内队列；callback 只推进每个 run 的 observed watermark，consumer 按
delivered watermark 从权威 store 以最多 64 条的 page 分批补读。所有 `LifecycleStore` 实现必须
支持 `limit`，并在复制或解码前执行上限。新 manager 不扫描、
恢复或 attach 既有 active/waiting lane；此时新的 idle submit 会由 store 的 session-lane CAS 拒绝。
Durable run、history、checkpoint、interaction、cancellation、result 和 `RunEvent` 始终由 runner/store
权威负责。

## Managed 组合钩子（包内）

`AgentRunner._start_managed()` 与 `_resume_managed()` 是供 `iris.harness` 内部组合层使用的
package-private hooks，不是 `iris.harness` 导出。它们不改变 complete-run 语义：coroutine 仍等待
waiting 或 terminal `RunResult`，public `start()` / `resume()` 只是使用空 hook 委托给它们；
`recover()` 没有 managed 变体。

Managed 调用可注入 activation-scoped steering port、同步 durable event callback 和
`asyncio.Event` admission signal。Signal 只会在 create/resume durable mutation 成功、对应 events
已 relay 且 exact activation 注册进 runner `_active` 后置位；立即 terminal 或 mutation/registration
失败不会产生虚假 signal。

Store-backed commit port 与 runner-owned create/resolve/begin/cancel/finish mutation 只在成功后把
新的 durable `RunEvent` 同步 relay。Callback 异常只记录日志，不回滚 mutation 或改变 `RunResult`。
公开 `RunEventObserver` 签名不变：同一 observer 内按 run sequence 串行保序，不同 observer lane
并行；每个 event 默认最多等待 30 秒，可用 `observer_event_timeout_s` 覆盖。Timeout 或普通异常只
记录 warning 并继续，不改变 durable result；同步 callback 不是新的 public observer registry。

每次 activation 中，runner 与 commit port 共享私有 `_RunEventCollector`，由它唯一持有累计
事件与 `(run_id, sequence)` 去重键。新增批次只检查本批事件；取消事件即使被两条路径观察，
同步 callback 也只对首次收集执行一次。Callback 失败不阻断后续 live publisher。

## Cancellation 与 recovery

命令环境由 root 拥有，一个服务覆盖其 session 与 child。正常 COMPLETED/WAITING 保留环境；
非正常结束先结算关联 child，再停止并排空命令调用，最后 `FinishRun` 释放 session lane。
Docker 停整个共享容器，其他独立 run 的模型/原生工具/HITL 继续；Native 只覆盖目标 session
的当前命令。当前调用已有停止收据时只等待那一轮排空，不再停止已重启的环境。

`PendingSettlement` 只在进程内保留原 outcome/error、typed target 和当前 receipt。并发调用
共用一个“停止→排空→终态”任务；取消某个等待者不会取消它。`IrisCommandCleanupError`
保留 ACTIVE/WAITING 和 lane，直接调用者收到异常，并通过 root 发布 `CommandCleanupFailed`
小型 live fact；没有 publisher 时记录错误。下一次 cancel/recover/resume 优先只重试原结算，
不改写原失败原因或重跑模型/命令。已知工具结果在清理错误传播前提交，不倒退成未知 claim。
Middleware 后处理期间取消又遇到清理失败时，Runtime 会同时交回原停止原因和进程内清理错误；
pending 保留原取消或 deadline 意图。SDK task 取消则保留 cleanup-only 意图，不改写成 FAILED。

总期限 timer 由 root 持有，跨 WAITING 和临时 child runner 关闭继续有效；child 由 typed route
重建，终态撤销 timer。root close 先撤销尚未触发 timer，等待已触发结算和 pending，再关闭资源。
期限已经过期的 start 仍先取得 ACTIVE/fence/lane，但不提交输入；预算拒绝不写终态。
两种路径都由 runner 清理后再结束，不能通过 store 捷径越过异步清理。

结算失败时，runner 使用注入的 Clock 核对 absolute deadline，不依赖 timer 是否已经获得调度。
到期后的 provider 异常、`response.failed` 和 provider 取消收尾错误结算为 `DEADLINE_EXCEEDED`；
到期前的 provider 错误保持 `FAILED`。未提交的工具 claim 仍优先结算为
`OUTCOME_UNKNOWN`。

`cancellation_requested` 是 durable fact，不等于已取消。Runner 先持久化请求，再发送本地
signal；存在 claim 时不整体取消 activation task。`ToolExecutor` 将 signal 转为普通 async
callable、自定义异步 `BaseTool` 或 THREAD callable 的 body task 取消，也中断仍在等待的 Middleware
包装链，并等待已开始的操作收口。压住 `CancelledError` 的协程及 INLINE 阻塞仍可能延迟 settlement；`cancel()`
只等待 durable terminal result，不提前返回 cancelled。

body 已完成，或响应 signal 取消后仍正常返回时，结果经过后处理并按既有顺序 durable commit 后再结算
cancelled。有限本地文件 IO 和 artifact 作业会在取消后收回确定结果，先提交工具事实再响应
task cancellation、timeout 或 sibling cancellation；并行结果仍只提交无空洞的 ordinal 前缀。
没有 signal 的外层 task cancellation 先完成命令环境和关联 child 清理，再传播，留下可恢复的
ACTIVE 事实，不冒充用户取消。重复 task.cancel 不提前跳过这项收尾；失败保留 cleanup-only pending。
未结算 claim 仍使 run 以 `TOOL_OUTCOME_UNKNOWN` 收口，包括只读调用；
自定义 THREAD callable 的 worker 可以继续运行，晚到返回不能改写 durable result、history、checkpoint 或 events。

Runner 的 live signal 与 store-backed commit port 使用
`iris.exceptions.IrisCancellationRequestedError` 通知 runtime 协作式收口；该类型不属于
`iris.tools` 公共错误面。

Store-backed commit port 在每个 effect/commit 安全边界重新读取最小 run control，不跨边界缓存。
它只接受 control 完全相等，或同一 active activation 上 revision/event sequence 各推进一步且由唯一
`run.cancellation_requested` event 证明的取消；phase、fence、跳号、重复取消或 event/payload 不匹配
全部 fail closed。随后 mutation 仍以原有 revision 与 activation CAS 为最终授权。

runtime 的只读并发窗口使用固定内部上限 8；它不增加 public config/schema/API。窗口中每个
调用都有独立 durable claim，body 可以乱序结束，但只有连续的已知 result prefix 会按 ordinal
进入 history/cursor/checkpoint。claim telemetry 的 event 顺序不是 ordinal 契约。任一未提交
claim 都会使 cancellation、deadline 或程序中断结算为 outcome unknown；现有 terminal
settlement 会在同一 aggregate transaction 中关闭该 activation 的全部 unresolved claims。

Todo 的自查目标保存在 v4 cursor 的 `todo_reminder_step`，新 Run 显式初始化为 None。
恢复到目标步骤时重新读取当前文件并复用已有 pending model reservation；目标响应提交后
继续工具/HITL 不重复自查，outcome_ready 只结算。宿主最终取得实际最后一条 assistant 回复，
流式输出中此前的候选回复仍可能已经可见。

active recovery 会验证 checkpoint v4、session revision、usage counters
与 cursor。只要存在 unresolved claims 就不会重放工具；recovery 原子 abandon 旧 activation，
取得新 RECOVER fence 并保存 BLOCKED_UNKNOWN checkpoint，清理后再关闭 claims 并写 unknown 终态。
正常 parent/control/
infrastructure 退出会先等待 runtime children drain，随后 revoke commit port；不会允许迟到 child
继续写入。同步阻塞 callable 不保证并发加速，并且仍可能延迟 settlement。

恢复使用当前 runner 的模型、context、工具和权限配置。修改 system prompt、压缩预算或
工具目录不会触发全局配置相等检查。已保存的请求、运行限制、cursor、调用身份和执行结果
仍来自原 run；待执行工具仍需满足参数与当前权限规则。运行记录和 checkpoint 不保存环境指纹。

Context 和 runtime 各自使用共享的 [`iris.utils.TemplateRenderer`](../utils/README.md)，
保留 Jinja 原生按需加载、编译缓存与默认更新检测，同一 runtime 的后续渲染可读到文件修改。
模板默认关闭自动转义，XML 模板自行声明转义；`StrictUndefined` 在渲染时检查，context
字符上限在完整文本生成后检查。详见 [`iris.context`](../context/README.md)。

start、resume、subagent parent resume 和 recover 都从 durable run 传递 `run_input` 与
`initial_session_message_count`。新 run 在 `before_input` 将 BCI 和用户输入归档；session 首次
输入还会把实际选定的 `SessionContextWindow` 与输入/checkpoint 原子提交，再进入 `before_model`。
即使没有消息增量，窗口初始化也推进一次 session revision；provider 失败不撤销已提交输入。

有效 runtime memory service 存在时，工具循环、后续 run、HITL 与恢复重放已提交的概览窗口，
不重新查询或加载更新后的文件。
成功压缩时新摘要和新窗口同事务替换，失败时保留原状态；fork 的目标窗口未初始化，首次输入
重新采用。窗口文本不另存一份到 checkpoint。lifecycle SQLite 使用 schema 11，checkpoint 为 4，
旧库/旧 checkpoint 按既有边界拒绝，不迁移或自动删除数据。

新 runtime 未绑定 memory service 时，普通请求、HITL 与恢复都不把已保存概览追加到 system。
这个选择不修改已保存窗口，也不额外推进 session revision；静态 memory 和普通历史仍保留。
关闭后的成功压缩在原事务内提交空窗口；失败保留旧摘要和窗口，后续请求继续屏蔽该概览。

`memory.enabled` 默认关闭；开启时自动注册 Search/Fetch，写工具仍需显式声明。关闭时不挂载
宿主传入的 service；开启时复用其原有依赖。改配置后重建 Agent 并使用新会话，不支持热切换。
记忆读取由模型自主调用 Search/Fetch，结果是普通工具历史，可正常压缩，没有跨轮已读表或
特殊原文保护。概览指导主题范围，所有 namespace 合计占可用输入预算的2%；没有概览时正常
聊天但暂不使用长期记忆。静态 `context.yaml` memory 仍为独立固定上下文，不进入会话历史。
BCI、原始用户输入和最新 steer 保持既有保护；压缩提交后的恢复使用新 revision 和同一 pending 步。

`memory.generation.enabled: true` 开启 root runner 的自动取材与维护，默认关闭。
宿主显式创建一个 [`MaintenanceCoordinator`](maintenance.py)，多个 root runner 通过
`bind_maintenance()` 借用它。`from_config*` 不创建私人维护循环；启用功能却没有绑定时，
首次准备或运行会报告配置错误。协调器的 `idle_seconds` 来自宿主选定的
`config.maintenance.idle_seconds`，默认 300 秒。
未启用 Memory 的 runner 也可调用 `runner.bind_maintenance(coordinator)`，只贡献前台状态，
使同宿主其他 runner 的维护及时让位；不会创建 Memory 资源或捕获材料。

宿主先构造完整的 `MemoryService`（flush/dream、overview provider/model 和 mirror），
把同一实例注入 runner 和 `MemoryMaintenanceBinding`，并传入实际 SQLite 数据库路径、
write namespace。注入服务保留自己的 provider、预算和 IO 模式。以下函数展示共享装配与关闭：

```python
from pathlib import Path

from iris.agents import AgentConfig
from iris.harness import AgentRunner, MaintenanceCoordinator, MemoryMaintenanceBinding
from iris.lifecycle import AgentRunRequest
from iris.memory import MemoryService
from iris.providers import CompletionProvider


async def run_sessions(
    config: AgentConfig,
    provider: CompletionProvider,
    memory: MemoryService,
    database_path: Path,
) -> None:
    coordinator = MaintenanceCoordinator(idle_seconds=config.maintenance.idle_seconds)
    binding = MemoryMaintenanceBinding(
        service=memory,
        database_path=database_path,
        namespace=config.memory.write_namespace,
    )
    runners = [
        AgentRunner.from_config(config, provider=provider, memory_service=memory)
        for _ in range(2)
    ]
    try:
        for runner in runners:
            runner.bind_maintenance(coordinator, memory=binding)
        for index, runner in enumerate(runners):
            await runner.start(AgentRunRequest(input="处理项目任务", session_id=f"session-{index}"))
    finally:
        for runner in runners:
            await runner.aclose()
        await coordinator.aclose()
```

[`_capture.py`](_capture.py) 只保存原始材料，独立于维护调度。
root Run 首条消息提交前登记 lifecycle `source_id`、run 和 session；压缩提示、WAITING/终态退出
及关闭保存尚未捕获的后缀。每页至多 128 条消息，逐页提交，只到达完整终态计数后封源。
WAITING 不封源；完成、失败、取消及无 runtime 的恢复终态保留真实 outcome。
BCI、system/reasoning 和记忆读回正文不成为新证据；Search/Fetch 保留条目引用，工具调用与结果
保留稳定 call_id。持久捕获水位避免重复经历。

自动学习只消费终态且完整捕获的 Run。某 session WAITING 时，其旧终态材料也排除，其他会话
仍可维护。Memory 工具的变更先通过 call_id 关联捕获来源；尚未关联时保持 pending。
每次消费提交前重读生命周期资格；缺少 reader 时保留 pending。SQLite lifecycle 支持重启续作，
纯内存 lifecycle 丢失后不自动猜测旧材料资格。

一个宿主最多一项 Memory 维护。规范化实际 DB 路径和 namespace 确定原生 OS 锁；锁内重读、
执行有界领域周期并排空真实 IO，同库同 namespace 的独立进程不会重复调用模型。
锁忙让出本地位置，至少等待 `max(idle_seconds, 1 秒)` 后自动再试，多资源按有界周期轮转。
Memory 服务拥有 dream 优先、flush 后 dream、投影与 overview 修复的顺序，协调器不解释内容。

前台 admission/activation 全部退出并安静达到 idle 后才维护。新前台立即撤销未提交生成，
不等待模型或资源锁；Goal/follow-up 的短交接保留同一前台计数。
THREAD 服务使用专用 worker，INLINE 保持调用线程执行；已派发同步工作真实退出前保留锁和任务位置。
模型失败等待外部新活动或重启，自身维护写入不构成新活动；no-change 正常推进消费位置。

`runner.aclose()` 只排空自己的捕获、解除借用并关闭原有自有资源，不关闭共享协调器、服务或 reader，
也不受其他 runner 前台计数阻碍。关闭所有借用该资源的 runner 后，宿主可调用
`await coordinator.unbind_memory(binding)` 单独撤销资源；它会取消并排空该资源维护，其他资源继续。
宿主退出时 `await coordinator.aclose()` 停止并排空维护，随后自行关闭共享 provider/service/store；
这些关闭路径都不会临时补跑学习模型。维护 usage 独立于 `RunUsage`，新概览仍在新窗口或成功压缩时采用。

`RunUsage` 的 input/output/total 只统计主模型；摘要调用累计在 `usage.compaction`，总消耗由两者
逐字段相加。摘要 usage-only 提交只推进 run revision，不产生 durable event；commit port 立即
接受新 revision，使后续取消读取保持有效。投影提交独立产生 `context.compacted`，原文历史保留。

## 公开接口

`iris.harness` 导出 `AgentRunner`、`MaintenanceCoordinator`、`MemoryMaintenanceBinding`、
`SessionHistory`、`SessionManager`、`SubmitReceipt`、
`ResumeReceipt`、`SubmissionEvent`、
`SessionSubmissionEvent`、`SessionEvent`、`LiveFact`、`LivePublisher`，以及 run
request/options/limits/runtime options、phase/stop reason/usage/error/snapshot/result 和 run
events/observer。Store commands 仍属于 `iris.lifecycle`。

## 验证

```bash
uv run pytest tests/harness
uv run ruff check src/iris/harness tests/harness
uv run mypy src/iris/harness
```
