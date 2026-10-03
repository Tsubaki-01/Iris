# Decision：独立的语义判断 SDK

`iris.decision` 为封闭选择、命题概率和分级评分提供异步接口。它不生成聊天回复，也不替代
`CompletionProvider`。业务负责准备问题及解释答案；本包负责公共类型、Jev HTTP 协议和连接。

## 独立调用

通过既有配置提供 `IRIS_PROVIDER_API_KEYS__TYPESAFE`，然后调用一次共享状态上的多个问题：

```python
import asyncio

from iris.config import init_config
from iris.decision import ChoiceQuestion, DecisionRequest, JevClient, ScoreQuestion


async def main() -> None:
    """一次请求同时完成选择与评分。"""
    config = init_config()
    async with JevClient(api_key=config.provider_api_keys["typesafe"]) as client:
        response = await client.evaluate(
            DecisionRequest(
                state={"query": "读取说明", "tools": {"read": "读取文件", "edit": "修改文件"}},
                questions={
                    "tool": ChoiceQuestion(
                        instructions="为 state.query 选择 state.tools 中最合适的工具。",
                        options={"read": None, "edit": None, "none": "没有匹配工具"},
                    ),
                    "relevance": ScoreQuestion(
                        instructions="评价 read 工具对 state.query 的适用程度。",
                        levels=("无关", "部分适用", "直接适用"),
                    ),
                },
            )
        )
        print(response.answers["tool"])
        print(response.answers["relevance"])
        print(response.model, response.usage)


asyncio.run(main())
```

宿主已有配置时使用已有的 `get_config()`，不要重复初始化全局配置。

## 输入、输出与边界

- `ChoiceQuestion` 从 `options` 选一个 ID；多个独立意图可在同一请求放多道题。
- `BooleanQuestion` 返回命题成立概率，厂商 wire 类型为 `noul`。
- `ScoreQuestion` 按有序 `levels` 返回期望档位；它不是正确率或校准后的匹配概率。
- `DecisionRequest` 是原始 SDK 输入的校验边界，包含一个 JSON `state` 和非空问题字典。
  后端 `model` 在 client 构造时指定，不放进业务请求。未知输入字段拒绝。
- `DecisionResponse.answers` 按原问题 ID 映射；结果保留实际 provider/model、答案分布及独立用量。
  回答顺序不依赖 JSON map 顺序；Score 保留服务值，不通过显示概率重算覆盖。

`DecisionEvaluator` 只要求 `async evaluate(request)`。宿主替身、工具与其它消费者直接使用已解析
的类型化结果，不探测替身方法、不重复验证。Jev 的选项上限、响应结构与题目对应关系由 HTTP
边界校验；业务匹配规则留在业务模块。

## Jev 连接与错误

默认模型为 `jev-1.13.0`，总期限 5 秒，端点为 `https://api.typesafe.ai/v1/systemone`。
构造不联网，首次调用才建立自有 HTTP client；一次 evaluate 只发一次 POST，无自动重试、
分片、低置信重问或本地降级。Choice 最多 255 选项，Score 为 2–10 档。

`JevClient.aclose()` 或 async context manager 关闭自有资源。客户端不接收外部 HTTP client 的
所有权移交；借用 evaluator 的业务不负责关闭连接。网络、HTTP、期限和解析失败统一抛出
`IrisDecisionError`，沿 provider source / `DECISION_ERROR`；外层取消原样传播。凭据只用于请求头。

## Agent 配置与接点

在 `agent.yaml` 引用独立配置；路径相对 Agent YAML 解析。直接传入 `AgentConfig` 时以
`config_path` 所在目录为基准，未提供则使用当前目录。

```yaml
# agent.yaml
decision:
  path: decision.yaml
context_policy:
  enabled: true
  deferred_tools: true
memory:
  enabled: true
```

```yaml
# decision.yaml
provider: typesafe
model: jev-1.13.0
timeout_seconds: 5
tools:
  discovery: true
memory:
  recall: true
```

省略 `decision` 或关闭全部接点时不构造客户端、不要求 key。开启工具发现必须同时启用
deferred tools；开启记忆召回要求 `memory.enabled=true`。两开关独立，只开启记忆也会构造
客户端。未知配置字段或依赖冲突在装配时报告 `IrisConfigError`。配置固定于构造期。

`tool_search` 的两种后端共用 `queries` / `include_groups` 输入。每个意图至多选一项；本地
逐意图取词法 top-1，增强模式将全部允许的 deferred 候选一次批量 Choice，eager 工具仍直接
调用。出站只包含 queries 与候选 name/description，失败不回退。详见 [工具说明](../tools/README.md)。

`memory_search` 的两种后端共用 `query`、`required_terms`、`categories`、`kinds`、`limit`。
增强模式读取全部允许 ACTIVE 条目，在本地匹配显式必要词组后一次批量 Score，不按 query
先做词法搜索或 top-k。出站只有 query 与完整正文数组；评分达到 2 后稳定降序，最后应用
limit，返回全文。`has_more` 表示达标结果数超过 limit，不提供分页。见 [记忆说明](../memory/README.md)。
evaluator 只注入本 Agent 的 Search 工具，两个 Agent 共用 MemoryService 时仍可各选模式。

`AgentRunner.from_config/from_config_path` 和 `RuntimeFactory.from_config/from_config_path`
都接收 `decision_client=`。注入对象只需实现 `evaluate`，不读取 TypeSafe key，也不由 Iris 关闭。
自建客户端保存在 runtime environment，同一 Agent 跨 session/Run 复用，由环境最终关闭；
child 按自己的配置独立构造，不继承 root 注入。增强的内置工具声明 READ+NETWORK，默认
权限允许，自定义 policy 与执行前权限刷新仍生效。

## 维护入口

公共输入和可信结果见 [models.py](models.py)，窄协议见 [client.py](client.py)，厂商映射与连接见
[jev.py](jev.py)，配置与客户端装配见 [config.py](config.py) 和 [factory.py](factory.py)。
测试使用 MockTransport 在本地验证请求和响应契约。

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run pytest -p no:cacheprovider --basetemp="$PWD\tmp\pytest-tmp" tests/decision
```
