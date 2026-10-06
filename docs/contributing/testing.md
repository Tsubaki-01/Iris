# 测试、示例与研究实验

验证应回答本次改动带来的问题。确定性测试适合检查契约与状态转换，真实模型示例适合确认服务集成，评测则需要明确题集与评价目标。它们的成功不能相互替代。

## 运行相关测试

在仓库根目录运行相关测试：

```shell
uv run pytest tests/harness/test_session_history.py
```

替换测试路径即可验证自己的改动。

## 从改动找到测试

| 改动 | 主要测试入口 |
| --- | --- |
| Agent YAML / 配置 | [tests/agents](../../tests/agents) |
| 上下文 / 压缩 | [tests/context](../../tests/context)、[tests/runtime](../../tests/runtime) 的 context/compaction 用例 |
| Runner / 恢复 / SessionManager / 历史 | [tests/harness](../../tests/harness) |
| 生命周期与存储 | [存储契约测试](../../tests/store/test_lifecycle_store_contract.py)、[tests/store](../../tests/store)及 harness 的状态转换测试 |
| 工具 / MCP / 命令 | [tests/tools](../../tests/tools)、[tests/mcp](../../tests/mcp)、[tests/command](../../tests/command) |
| 消息 / 模型 / 图片 | [tests/message](../../tests/message)、[tests/providers](../../tests/providers)、[图片处理](../../tests/utils/test_images.py) |
| 记忆 / 经验 / 目标 / 清单 | [tests/memory](../../tests/memory)、[tests/evolution](../../tests/evolution)、[tests/goal](../../tests/goal)、[tests/todo](../../tests/todo)及 harness 的相关集成测试 |
| 流式 / 观测 / 语音 | [tests/streaming](../../tests/streaming)、[tests/observability](../../tests/observability)、[tests/speech](../../tests/speech) |

先定位现有用例和复用的 provider/store 替身，再补一个能复现新需求的测试。不要为了一个消息映射错误搭起完整 UI、远端模型或多服务集成。

代码风格和类型配置在 [pyproject.toml](../../pyproject.toml)。例如修改一个 Python 文件后，可针对它运行 `uv run ruff check 路径` 与 `uv run mypy 路径`。`.pre-commit-config.yaml` 定义已有提交检查，不需要为本次文档建设新增一套门禁。

## 可运行示例的证据范围

| 方式 | 能证明什么 | 不能单独证明什么 |
| --- | --- | --- |
| 配置解析、导入或代码块编译 | 字段和语法可接受 | 真实功能完成 |
| 脚本化 provider + 真实 Runner/工具 | 接口接线、状态转换和本地副作用符合脚本 | 模型会自主选择正确行动 |
| 真实 provider 示例 | 某次环境中的协议、凭据和任务流程可运行 | 全部模型、长期稳定性或普遍任务成功率 |
| 明确题集的评测 | 所定义样本与指标上的表现 | 超出题集范围的能力 |

文档里的完整例子应交代凭据、文件、可选服务和工作目录。输出不确定时描述成功条件；不要用固定模型文本作为验收断言。

## Inspect AI 与检索实验

[evals](../../evals/README.md) 提供仓库级 `iris_solver()`，让 Inspect 调用真实 AgentRunner、投影结果与用量并收尾资源。它不随 Iris Python 包发布，使用时在仓库根目录安装 `eval` 依赖组：

```shell
uv sync --group eval
```

当前 Inspect 接入本身不附带完整题集、benchmark adapter 或评分器。另有独立 Jev 工具检索实验；它们的候选目录、题集、调用费用和评价范围在各实验说明中维护，不作为默认 Agent 的效果保证。

增加评测时先明确想测的行为、固定输入与预算、结果记录和评分方式。报告实际运行条件，区分检索选对工具、工具执行正确与最终任务完成。

下一步：[贡献流程](index.md) · [文档维护](docs.md)。
