# `iris.utils`

`iris.utils` 提供跨领域共享的基础工具。包入口导出 `TemplateRenderer`，用于从 Jinja2 文件生成
文本；`iris.utils.images` 提供图片处理与文件副本保存。调用方可直接使用这些工具，无需通过
其他领域模块取得能力。

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

默认构造器不接收参数。`render_file(template_path: Path, context: dict[str, Any]) -> str` 接收入口
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

## 固定操作内的模板源

需要在完整操作内固定正文时，使用 `TemplateRenderer.freeze_directories(directories)`：

```python
renderer = TemplateRenderer.freeze_directories([Path("prompts")])
text = renderer.render_file(Path("prompts/greeting.j2"), {"name": "Iris"})
```

工厂捕获各入口父目录下全部可加载文件内容，目录之间相互隔离；返回的 renderer 只从内存
取源，后续磁盘改动不影响当前实例。动态依赖、候选列表和可选文件的缺失状态一并固定，实际
使用时才解码与编译，不提前编译未执行分支中的模板。每次渲染仍接收当前变量，Jinja 输出
规则与默认文件 renderer 相同。捕获不承诺多文件事务原子性。

命名项目模板由 [`iris.prompts`](../prompts/README.md) 负责初始化和选择快照时机；runtime
装配也使用冻结 renderer 固定自定义 system/context 模板。独立 `TemplateRenderer()` 继续
按文件更新检测，不因新增快照能力改变语义。

读取、解码、解析或执行模板失败时抛出 `IrisTemplateError`，包含模板路径和底层错误。
业务调用边界负责转换领域异常：context 与 compaction 使用 `IrisContextError`，memory 使用
`IrisMemoryError`，skill 使用 `IrisSkillError`，goal 使用 `IrisGoalError`。

## 内置 prompt 与调用方

内置任务和上下文指令集中放在 [`../prompts/`](../prompts/)；Python 继续准备模型输入和 schema。
每个模板顶部用 Jinja `{# ... #}` 注释说明用途、调用位置和变量；这些注释不会进入渲染结果。

| 调用方 | 模板 | 实例生命周期 |
| --- | --- | --- |
| `ContextBuilder` | 用户配置的 context 模板 | Builder 持有，可通过 `template_renderer` 注入 |
| Runtime compaction | `compaction.j2`、`compaction_input.j2` | 一次完整压缩的项目来源快照 |
| Runtime 记忆窗口 | `memory_context.j2` | runner/runtime 构造期项目来源快照 |
| `MemoryService` | `memory_flush.j2`、`memory_dream.j2`、`memory_overview.j2` | 整个自动 cycle 或单次独立生成操作共用快照 |
| `SkillCatalog` | `skill_catalog_usage.j2` | 构造期来源，后续复用使用指引 |
| Goal、Todo 与 Decision | 对应项目命名模板 | runner/runtime 构造期快照，领域变量保持动态 |

实现位于 [`templating.py`](templating.py)，公共导出位于 [`__init__.py`](__init__.py)。
模板加载、更新、转义与异常契约由 `tests/utils/test_templating.py` 验证；各领域测试覆盖请求装配
和领域异常转换。

## 准备和保存图片

[`images.py`](images.py) 接收原始图片字节或明确的文件路径，不持有 session、消息模型或
provider 格式。调用方负责选定 session 的 image-cache 目录，并将返回信息投影为消息块。

```python
from pathlib import Path

from iris.utils.images import save_image

saved = save_image(Path("photo.png"), cache_dir=Path("chosen-image-cache"))
print(saved.original.path)  # 保留输入字节的绝对路径
print(saved.model.path, saved.model.mime_type)  # 模型应读取的文件及实际 MIME
```

- `prepare_image(data: bytes) -> PreparedImage` 一次解码静态 PNG、JPEG 或 WebP；返回包含
  字节、实际 MIME、宽高的 `original` 和 `model`，不产生 base64。
- 小图无需方向修正、尺寸和字节均合适时保留原编码，不放大。否则按 EXIF 修正方向，等比
  缩至宽高各不超过 2000px，模型版最多 3.75MiB。PNG/透明图先无损 PNG；不透明图可依次
  尝试 JPEG 质量 85、70、50、30。仍超限时最多再减半两次，每个候选由原始解码图生成。
- 透明图不转 JPEG、不填充背景、不做 palette 量化；编码后的实际 MIME 写入结果。上述限制
  是 Iris 的客户端处理策略，不代表所有 provider 的限制，也不等于视觉 token 计费规则。
- `save_image(source: Path | bytes, *, cache_dir: Path, reuse_source=False) -> SavedImage` 保存同一次读取的快照；
  路径来源分块读取，文件全部关闭后返回 `SavedImageFile` 信息。每次导入使用新的随机资产 ID；
  有变换时保留 original/model 两份文件，无变换时两个引用指向同一份文件。
- 工具调用方确认源路径属于已有 image-cache 后可使用 `reuse_source=True`：仍完整解码处理，
  合规则两个引用复用该文件；需变换则保留原文件，仅向目标目录写新模型版。普通 SDK 导入仍保存新快照。
- 解码、处理、读取或写入失败抛出 `IrisImageError`；有限候选用尽不返回超限图。写入失败只
  清理本次创建的半成品，不影响已有文件。正常完成后不自动删除副本。

图片处理和文件契约由 `tests/utils/test_images.py` 验证。`PreparedImage`、`SavedImage` 及其
子对象是进程内不可变 DTO，不负责 JSON 持久化或 session 生命周期。
