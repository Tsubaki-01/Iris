# `iris.utils`

`iris.utils` 提供跨领域共享的基础工具，目前导出 `TemplateRenderer`，用于从 Jinja2 文件生成
文本。Context、runtime、memory 和 skill 可直接使用它，无需通过其他领域模块取得渲染器。

## 使用文件模板

创建 `prompts/greeting.j2`：

```jinja2
你好，{{ name }}。
```

在包含 `prompts/` 的目录运行：

```python
from pathlib import Path

from iris.utils import TemplateRenderer

renderer = TemplateRenderer()
text = renderer.render_file(Path("prompts/greeting.j2"), {"name": "Iris"})
print(text)  # 你好，Iris。
```

构造器不接收参数。`render_file(template_path: Path, context: dict[str, Any]) -> str` 接收入口
路径和当前变量，返回 Jinja 的渲染结果。调用方负责模板路径、数据准备、消息角色、预算和实例
生命周期；同一个 renderer 实例可渲染多个目录中的模板。

## 渲染契约

- 默认 `autoescape=False`，纯文本、JSON 和 Markdown 中的 `<>&` 与引号保持原文。生成 XML
  时，在模板中使用 `{% autoescape true %}...{% endautoescape %}`，或对变量使用 `|e`。
- 使用 `StrictUndefined`，读取未提供的变量会失败。
- 使用 `trim_blocks=True`、`lstrip_blocks=True` 和 Jinja 默认尾换行规则。渲染器不额外
  `.strip()`；例如 context 和 compaction 由各自调用方去除首尾空白。
- 按解析后的入口目录复用 `Environment` 与 `FileSystemLoader`，每次调用 `get_template()`。
  Jinja 的编译缓存和默认更新检测按文件 mtime 处理入口及实际使用依赖的变更；最终输出不缓存。
- `include`、`import` 和 `extends` 使用 Jinja 原生语义，支持动态文件名和按需加载。

读取、解码、解析或执行模板失败时抛出 `IrisTemplateError`，包含模板路径和底层错误。
业务调用边界负责转换领域异常：context 与 compaction 使用 `IrisContextError`，memory 使用
`IrisMemoryError`，skill 使用 `IrisSkillError`。

## 内置 prompt 与调用方

内置任务和上下文指令集中放在 [`../prompts/`](../prompts/)；Python 继续准备模型输入和 schema。

| 调用方 | 模板 | 实例生命周期 |
| --- | --- | --- |
| `ContextBuilder` | 用户配置的 context 模板 | Builder 持有，可通过 `template_renderer` 注入 |
| Runtime compaction | `compaction.j2`、`compaction_input.j2` | `RuntimeEnvironment.prompt_renderer` |
| Runtime 记忆窗口 | `memory_context.j2` | 同一 `RuntimeEnvironment.prompt_renderer` |
| `MemoryService` | `memory_flush.j2`、`memory_dream.j2`、`memory_overview.j2` | Service 自己持有 `prompt_renderer` |
| `SkillCatalog` | `skill_catalog_usage.j2` | 构造时读取一次，后续复用使用指引 |

实现位于 [`templating.py`](templating.py)，公共导出位于 [`__init__.py`](__init__.py)。
模板加载、更新、转义与异常契约由 `tests/utils/test_templating.py` 验证；各领域测试覆盖请求装配
和领域异常转换。
