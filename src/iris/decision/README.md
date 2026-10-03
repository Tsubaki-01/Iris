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

## 维护入口

公共输入和可信结果见 [models.py](models.py)，窄协议见 [client.py](client.py)，厂商映射与连接见
[jev.py](jev.py)。测试使用 MockTransport 在本地验证请求和响应契约。

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run pytest -p no:cacheprovider --basetemp="$PWD\tmp\pytest-tmp" tests/decision
```
