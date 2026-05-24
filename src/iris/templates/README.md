# iris.templates

`iris.templates` 提供官方 agent 配置模板的 scaffold 能力。它把 Iris 包内置的模板文件复制到
用户指定目录，供 CLI 或调用方复用。

当前官方模板只有 `file-agent`。

## 快速入门

```python
from iris.templates import scaffold_template

written = scaffold_template("file-agent", "./my-agent")
```

返回值是写入的目标文件路径列表。`file-agent` 会生成：

- `agent.yaml`
- `README.md`

生成的 `agent.yaml` 可以交给 `iris.agents.load_agent_config()` 读取。

## API

### `scaffold_template(template_name, target_dir, *, overwrite=False)`

从内置模板目录复制文件到 `target_dir`。

- `template_name`: 官方模板名称，例如 `file-agent`。
- `target_dir`: 目标目录。
- `overwrite`: 默认 `False`。目标文件已存在时会拒绝覆盖。

错误行为：

- 模板不存在时抛出 `IrisTemplateNotFoundError`，错误信息会包含可用模板。
- 目标文件已存在且 `overwrite=False` 时抛出 `IrisTemplateError`。

## 内置模板

### `file-agent`

`file-agent` 是最小本地文件助手模板，声明：

- OpenAI `gpt-4o-mini` 模型路由。
- 一个面向本地文件助手的 system prompt。
- 只读文件工具：`file.read`、`file.list`、`file.grep`。
- `writes: confirm`。
- `session.backend: none`。

模板只提供配置文件和说明文档，不实现 agent loop。

## 包发布要求

内置模板位于 `src/iris/templates/builtin/`。新增官方模板时，需要确保模板文件被打进
sdist/wheel，并至少包含：

- `agent.yaml`
- `README.md`
