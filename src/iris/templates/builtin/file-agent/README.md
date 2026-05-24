# file-agent

`file-agent` 是 Iris 的最小本地文件助手模板。它声明 OpenAI 模型路由、基础 system
prompt 和只读文件工具，适合作为个人本地 agent 配置的起点。

## 包含文件

- `agent.yaml`: agent 声明式配置。
- `README.md`: 模板说明。

## 默认能力

`agent.yaml` 默认启用以下内置工具：

- `file.read`
- `file.list`
- `file.grep`

它不会启用写入工具，也不会启用 SQLite session。

## 使用方式

```python
from iris.agents import build_tool_registry, load_agent_config

config = load_agent_config("agent.yaml")
registry = build_tool_registry(config.tools)
```

生成的配置可以作为后续 agent loop 的输入，但模板本身不实现 agent loop。

## 调整建议

按需修改 `agent.yaml`：

- 修改 `model.provider` 和 `model.name` 切换模型。
- 在 `tools.builtin` 中增加 `file.write` 或 `file.edit` 启用写入类工具。
- 将 `session.backend` 改为 `sqlite` 启用轻量本地 session。
