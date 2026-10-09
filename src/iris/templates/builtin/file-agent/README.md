[English](README.en.md)

# `file-agent`

`file-agent` 是 Iris 的最小本地文件助手模板。它声明 OpenAI 模型路由、基础 system
prompt 和只读文件工具，适合作为个人本地 agent 配置的起点。

## 包含文件

- `agent.yaml`: agent 声明式配置。
- `README.md`: 模板说明。
- `README.en.md`: 英文模板说明。

## 默认能力

`agent.yaml` 默认启用以下内置工具：

- `file.read`
- `file.list`
- `file.grep`

它不会启用写入工具，也不会启用 SQLite session。
模型默认使用 `responses` 协议；`permissions.workspace: .` 相对本 `agent.yaml` 所在目录解析。
`session.backend: none` 仍有进程内会话状态，退出后不保留历史。

## 使用方式

在已安装 Iris 的 Python 环境中，进入本目录，配置 `IRIS_PROVIDER_API_KEYS__OPENAI` 后启动：

```bash
iris chat agent.yaml
```

如果凭据保存在本目录的 `.env` 中，显式使用 `iris chat agent.yaml --env-file .env`。
以下 SDK 片段只加载配置并构建工具注册表，不调用模型：

```python
from iris.agents import build_tool_registry, load_agent_config

config = load_agent_config("agent.yaml")
registry = build_tool_registry(config.tools)
```

完整执行由 CLI 或 `iris.harness.AgentRunner.from_config_path("agent.yaml")` 负责；模板本身不实现
agent loop。SDK 宿主用完 runner 后需 `await runner.aclose()`。

## 调整建议

按需修改 `agent.yaml`：

- 修改 `model.provider` 和 `model.name` 切换模型。
- 在 `tools.builtin` 中增加 `file.write` 或 `file.edit` 启用写入类工具。
- 将 `session.backend` 改为 `sqlite` 启用轻量本地 session。

当前 `tests/` 中没有模板专用测试；修改此模板时应同步补充 scaffold 行为测试。
