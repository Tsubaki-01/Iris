# 同步 Callable 的执行与责任边界

本文说明 Iris 如何执行同步工具、为什么框架不自动将同步 callable offload 到线程池，以及工具
实现者在显式使用 `asyncio.to_thread()` 时需要承担的并发、取消和副作用责任。

文中区分三类信息：

- **当前实现**：可以由 active source 直接确认的行为；
- **工具规范**：基于当前执行模型，工具实现者应遵守的约束；
- **当前缺口**：现有公共 API 还不能完整表达的能力，不应描述为已经支持。

## 核心结论

1. Iris 在 event-loop 线程中直接调用同步 callable，不自动把它提交到线程池。
2. 同步 callable 会占用 event loop，因而可能阻塞其他工具、事件消费、steer、interrupt 和 run
   settlement。
3. 需要调用阻塞 API 的工具应显式提供异步包装，并在确认底层函数适合在线程中运行后使用
   `asyncio.to_thread()`。
4. 取消对 `asyncio.to_thread()` 的等待不会终止已经开始运行的工作线程；线程中的函数必须自行
   支持协作式停止。
5. Iris 负责在 effect 前记录 durable claim，并在结果无法证明时保守结算为
   `outcome_unknown`；工具实现者负责线程安全、资源隔离以及外部副作用的幂等、事务、补偿或
   reconciliation。

## 当前执行事实

当前工具调用链是：

```text
runtime
  -> ToolBridge.execute_prepared()
  -> ToolExecutor
  -> await tool.arun(...)
  -> CallableTool.arun()
  -> self.func(**kwargs)
```

`CallableTool.arun()` 直接调用注册的函数。只有函数返回 awaitable 时，Iris 才继续 `await` 该
结果；实现中没有使用 `asyncio.to_thread()` 或 `run_in_executor()`。因此，下面的工具会直接阻塞
event-loop 线程：

```python
import time


def blocking_tool() -> str:
    time.sleep(10)
    return "完成"
```

即使 runtime 已为多个只读工具创建 asyncio task，同步函数在返回前也不会让出 event loop。任务
存在并不等于函数体能够并行推进。

当前 runtime 只允许连续的 read-only + concurrency-safe 调用进入有界并发窗口。每个调用仍先
独立 durable claim，body 可以乱序完成，但 result、session history、cursor 和 checkpoint 只按
模型顺序提交。

## 为什么不自动 Offload

框架无法仅根据一个函数使用 `def` 定义，就判断它是否适合在线程池执行。同步 callable 可能是：

- 很快完成的本地操作，offload 只会增加线程调度与 context 切换开销；
- 适合在线程中等待的阻塞式文件、网络或第三方 SDK 调用；
- 受 GIL 影响的 Python CPU 密集计算，线程池未必带来吞吐加速；
- 只能在主线程使用的 UI、signal 或原生库接口；
- 使用线程绑定资源的代码，例如某些数据库 connection；
- 读写未加锁的全局缓存、单例或其他共享可变状态；
- 已经在内部管理线程、进程或连接池的库。

统一自动 offload 会隐式改变函数的执行线程、并发度、取消语义和资源访问方式。原本正确的同步
代码可能因此产生竞态或线程归属错误，工具超时或 run 取消后还可能出现“调用方已停止等待，但
工作线程仍继续产生副作用”的状态。

因此，执行线程的选择必须是显式决策：工具实现者掌握底层库、共享状态和 effect 语义，应由其
决定是否 offload；Iris 不猜测。

## 如何选择执行方式

| 工具工作负载 | 推荐方式 | 说明 |
| --- | --- | --- |
| 很快完成的同步操作 | 直接同步执行 | 避免不必要的线程调度开销 |
| 原生 async I/O | 直接实现 `async def` | 使用底层库的非阻塞 API |
| 可安全在线程中运行的阻塞 I/O | `async def` + `asyncio.to_thread()` | 保持 event loop 可响应 |
| Python CPU 密集计算 | 进程池、原生实现或重新拆分 | `to_thread()` 通常只能改善响应性，不能保证加速 |
| 主线程绑定或线程不安全调用 | 不应直接 offload | 改用专用执行边界，或保持串行并接受阻塞限制 |

推荐的显式包装方式如下：

```python
import asyncio


def blocking_read(path: str) -> str:
    return blocking_library.read(path)


async def read_tool(path: str) -> str:
    return await asyncio.to_thread(blocking_read, path)
```

这个包装只声明“不要阻塞 event loop”。它不声明执行会更快，也不自动提供线程安全、强制取消或
副作用回滚。

## 线程安全检查

在使用 `asyncio.to_thread()` 前，工具实现者至少需要确认：

- 底层库是否明确支持从工作线程调用；
- 两次调用是否可能同时修改同一个全局变量、缓存、单例或文件；
- 是否复用同一个数据库 connection、文件句柄、client 或 session；
- 是否依赖 `threading.local()` 或调用线程身份；
- 是否会在线程中直接操作 `asyncio.Lock`、`asyncio.Event`、`asyncio.Queue` 等 loop-bound 对象；
- 是否要求严格调用顺序；
- 并发失败时是否会留下部分完成的外部 effect。

如果工具不能安全并发，应隔离每次调用的资源、使用适当的同步原语，或将其声明为不支持并发。
自定义 `BaseTool` 可以覆写：

```python
from typing import Any


def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
    del params
    return False
```

能力标签也必须准确声明。WRITE、EXECUTE、NETWORK、MCP 或 AGENT 等潜在副作用能力不能伪装成
只读工具来获得并发执行。

## 取消与超时

`asyncio.to_thread()` 返回一个可等待对象，但 Python 不提供安全、通用的工作线程强杀机制。
取消外层 await 时，通常发生的是：

```text
event-loop task: 收到取消并停止等待
worker thread:    继续执行 blocking callable
external effect: 可能继续发生
```

强制终止一个可能正持有锁、执行事务或调用原生库的线程，会使进程内状态不可预测。因此，线程
中的长任务需要自行实现协作式停止，例如定期检查 `threading.Event`：

```python
import asyncio
import threading


def blocking_operation(stop: threading.Event) -> str:
    while not stop.is_set():
        perform_one_bounded_step()
    return "stopped"


async def cancellable_tool() -> str:
    stop = threading.Event()
    try:
        return await asyncio.to_thread(blocking_operation, stop)
    except asyncio.CancelledError:
        stop.set()
        raise
```

这仍然只是请求停止：

- blocking callable 必须主动、频繁地检查停止信号；
- 正在执行的单次阻塞调用可能无法及时响应；
- 已经发生的外部副作用不会自动撤销；
- coroutine 已传播取消时，工作线程可能仍需一段时间才能退出。

因此，工具不能把“await 已取消”等同于“effect 没有发生”。

## Durable Lifecycle 语义

Iris 在真实工具 effect 前写入 durable claim。正常完成时，工具 result 会被提交，然后 runtime 才
推进 history、cursor 和 checkpoint。

如果工具 claim 后发生取消或超时，而 Iris 无法证明 effect 是否完成，runtime 会保守形成
`outcome_unknown`，不会伪造 cancelled、failed 或成功结果。显式 offload 的工作线程即使稍后
返回，其结果也不能在原 activation 已收口后作为迟到成功写入。

`outcome_unknown` 只准确表达框架掌握的事实，不负责修复外部系统。可能产生副作用的工具应根据
业务风险提供以下一种或多种机制：

- 幂等键，允许调用方安全重试或查询同一次操作；
- 数据库事务或原子操作，避免部分提交；
- 补偿操作，撤销已知的部分 effect；
- reconciliation，通过外部 operation id 查询最终状态；
- 明确的人工处置流程，用于无法自动判定的结果。

## Context 与共享状态

`asyncio.to_thread()` 会复制当前 `contextvars.Context`，但不会自动迁移任意 thread-local 状态，
也不会让 asyncio 的 loop-bound 对象变成线程安全对象。

当前 `CallableTool.arun()` 不会把 `ToolExecutionContext` 传入原始 callable。需要自定义 context
处理的工具应使用自定义 `BaseTool`，并遵循以下原则：

1. 在 event-loop 线程中读取所需 context；
2. 只把不可变或线程安全的普通数据快照传给工作线程；
3. 在线程中完成阻塞操作并返回普通结果；
4. 回到 event loop 后再进行框架状态推进。

不要在线程中直接修改 runtime cursor、lifecycle store 状态或 loop-bound cancellation 对象。工具
内部共享资源的生命周期、锁和清理也属于工具实现责任。

## 责任划分

| 责任 | Iris | 工具实现者 |
| --- | --- | --- |
| 参数、权限和 policy 检查 | 执行统一预检 | 声明准确的 schema 与 capability |
| effect 前 durable claim | 负责 | 不绕过标准执行入口 |
| 是否使用线程池 | 不自动决定 | 根据底层操作显式决定 |
| callable 是否线程安全 | 不自动证明 | 负责判断、隔离或加锁 |
| 是否允许工具并发 | 按声明进行调度 | 必须准确声明并发能力 |
| asyncio context | 维护 event-loop 控制流 | 不在线程中误用 loop-bound 对象 |
| 共享资源 | 不自动添加业务锁 | 管理资源隔离、同步和清理 |
| 取消 | 发出协作式信号并维护 lifecycle 结果 | 让阻塞操作响应停止信号 |
| 强制终止工作线程 | 不保证 | 不得假设取消等于线程终止 |
| 外部副作用回滚 | 不实现业务补偿 | 提供幂等、事务、补偿或 reconciliation |
| 无法证明的执行结果 | 保守记录 `outcome_unknown` | 提供事后查询或人工处置能力 |

## 当前 API 缺口

`BaseTool.is_concurrency_safe()` 当前默认返回 `True`，`CallableTool` 也没有直接暴露
`concurrency_safe=False` 的注册参数。与此同时，没有潜在副作用 capability 的工具会被视为只读，
因而可能进入 runtime 的只读并发窗口。

因此，在现有 API 下：

- 使用 `asyncio.to_thread()` 且访问共享可变状态的普通 `CallableTool` 必须自行保证线程安全；
- 明确不能并发的工具应实现自定义 `BaseTool` 并覆写 `is_concurrency_safe()`；
- 不应把新增注册参数、自动线程池调度或强制线程取消描述为当前能力。

是否为 callable 注册入口增加显式并发声明，属于独立 API 设计问题，不在本文中预设实现方案。

## 工具作者检查清单

在提交一个显式 offload 的工具前，应能够回答：

1. 为什么该调用需要线程池，而不是原生 async API？
2. 底层 callable 是否允许从工作线程调用？
3. 多次调用重叠时，共享状态是否安全？
4. run 取消或超时后，工作线程会发生什么？
5. 外部 effect 已发生但结果丢失时，如何查询、重试或补偿？
6. 工具的 capability 和并发声明是否与真实行为一致？
7. 是否可能长期占满默认线程池，需要专用容量或其他执行边界？

任何一项无法回答时，都不应把自动或显式 offload 当作无语义变化的性能优化。
