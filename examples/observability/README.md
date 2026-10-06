# Observability 离线示例

这两个示例使用固定模型响应，通过公开 `AgentRunner` 执行真实工具、流式交付与人工交互恢复。
不需要模型 API key，也不安装 MLflow SDK。模型不会自主推理；示例用于检查观测接入和调用树。

先安装采集依赖并启动独立后端，完整说明见 [observability README](../../src/iris/observability/README.md)。
在 MLflow 页面创建 experiment 后取得它的 ID，再在项目根目录运行：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv sync --extra observability
uv run --extra observability python examples/observability/basic.py --experiment-id 1
uv run --extra observability python examples/observability/streaming_child.py --experiment-id 1
```

将 `1` 换成你创建的 experiment ID。示例通过 `init_config()` 设置完整 OTLP HTTP/protobuf
地址 `http://127.0.0.1:5000/v1/traces` 和 `x-mlflow-experiment-id` header。
root runner 持有自动创建的观测服务，在最后的 `aclose()` 中统一收口；每次 Run 不强制 flush。

默认工作目录为项目 `tmp/observability-<示例>-<随机标识>/`，终端会输出实际路径和 Run ID。
`--workspace` 可指定示例目录，其中会写入演示说明或 child 配置；不会在项目根目录放业务文件。

| 示例 | 真实运行流程 | 看板检查点 |
| --- | --- | --- |
| `basic.py` | 模型请求 → `read_file` 读取中文说明 → 模型回答 | 一个 activation 下有两个模型节点和一个工具节点；模型输入包含工具返回正文 |
| `streaming_child.py` | 父级流式委派 → child 提问并 WAITING → 宿主给出固定回答 → child 和父级分别继续；另起一次失败 Run | 初次 activation 和恢复 activation 各自结束；恢复 control 包含 child，父级后续 activation 在 control 结束后开始；失败模型节点是 ERROR |

第一个模型请求显式报告输入用量为 `0`，未报告输出用量；最终父级回答只报告输出用量 `6`。
流式失败只报告输出用量 `0`。在原 span 属性中检查这些已知字段，未报告字段保持缺失，
不要把看板自动计算的 total 当成完整账单。正常流式终态应为 `completed`，不是 `abandoned`。

正文默认开启，便于查看示例输入输出。下面的参数用于检查不同采集策略：

```powershell
uv run --extra observability python examples/observability/basic.py --experiment-id 1 --no-content
uv run --extra observability python examples/observability/basic.py --experiment-id 1 --max-content-chars 80
uv run --extra observability python examples/observability/basic.py --experiment-id 1 --disabled
```

`--no-content` 保留调用元数据；超限时完整标准内容属性被省略，预览在 `iris.content.*.preview`，
对应字段列入 `iris.content.truncated_fields`。`--disabled` 执行相同业务流程而不启用采集。
可用 `--endpoint` 指向其他完整 OTLP traces URL。若本机代理阻止访问回环地址，可在运行示例的
临时终端会话设置 `$env:NO_PROXY = "127.0.0.1,localhost"`；Iris 不另行解析代理环境变量。

异步 `run_example(workspace, observability=service)` 接受完整宿主服务，原样借用且不负责关闭它。
测试通过这一入口接入内存 exporter，验证实际父子关系、内容、用量、恢复顺序和失败状态；
不启动 MLflow。真实看板的接收与显示仍需单独确认。

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run --extra observability pytest -p no:cacheprovider --basetemp="$PWD\tmp\pytest-tmp" tests/examples/test_observability_examples.py
```

等待期间没有运行中的长 span；通过已有 session/run ID 关联前后 activation，不要求看板把它们
自动合成同一 trace。恢复 control 与 child 属于同一 trace，MLflow 的 Session 分组可能采用
child session；通过 Traces 的调用树和 span 上的 run/parent ID 查看完整关系，不能只依赖
父 Session 列表。示例没有运行后台 Memory/Evolution 维护，其观测流程见包级文档。
