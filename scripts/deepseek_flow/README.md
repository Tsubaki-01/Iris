# DeepSeek Flow

这个包承载 `scripts/deepseek_agent_flow.py` 的实现。入口脚本保持命令兼容，
这里按职责拆分真实 DeepSeek live integration 验证逻辑。

## 运行目标

DeepSeek flow 的目标是验证 Iris 在真实 DeepSeek API 下的端到端行为，包括 provider
创建、LiteLLM route、tool calling、runtime loop、context、memory、session 和工具错误路径。

脚本不使用 fake provider 证明 runtime 行为。需要观察请求结构时，使用
`RecordingRuntimeProvider` 包装真实 provider，记录 `LLMRequest` 后继续调用 DeepSeek。
除 `provider_smoke_live` 的 provider 直连检查和 `run_turn_live` 的 API 语义专项检查外，
业务链路验证都通过真实 `run_loop()` 执行。

## 模块职责

- `bootstrap.py`: 初始化脚本运行时的 repo root / `src` 导入路径。
- `catalog.py`: 场景目录，声明每个 live 场景所属模块、runtime API 和验证说明。
- `constants.py`: DeepSeek flow 常量和场景名称。
- `models.py`: `ScenarioReport`、`ScenarioRunner` 等共享类型。
- `config.py`: Iris 配置初始化、API key 解析、日志配置和错误脱敏。
- `reporting.py`: 场景报告结构、聚合报告和 Rich 输出。
- `providers.py`: `RecordingRuntimeProvider`、真实 provider 创建、provider smoke 场景。
- `fixtures.py`: 临时 workspace、`agent.yaml`、Python tool module 写入。
- `runtime_scenarios.py`: `run_loop`、`run_turn` API 语义、`context.yaml` live 场景。
- `memory_scenarios.py`: `memory_results` 和 `memory_query` live 场景。
- `session_scenarios.py`: SQLite session live 场景。
- `file_tool_scenarios.py`: 内置文件工具 live 场景。
- `python_tool_scenarios.py`: YAML 注册 Python 工具 live 场景。
- `tool_error_scenarios.py`: 工具错误路径 live 场景。
- `tool_scenarios.py`: 工具场景聚合导出。
- `runner.py`: 场景注册、选择、执行和异常归一化。
- `cli.py`: 命令行参数解析和脚本主入口。
- `__init__.py`: 面向入口脚本和测试的公开导出。

## Live 场景

默认 `--scenario all` 会运行以下场景：

- `provider_smoke_live`: 真实 DeepSeek 直连，严格断言输出 `IRIS_PROVIDER_OK`。
- `runtime_read_loop_live`: 真实 `run_loop()`，验证 `read_file`、tool result 回灌和最终回答。
- `run_turn_live`: 真实 `run_turn()`，验证一次 provider 调用和一次工具执行。
- `context_yaml_live`: 通过 `run_loop()` 验证结构化 `context.yaml` 的消息顺序和请求内容。
- `memory_results_live`: 通过 `run_loop()` 验证显式 `RuntimeOptions.memory_results` 注入。
- `memory_query_live`: 通过 `run_loop()` 验证 `MemoryService(SQLiteMemoryStore)` 写入和召回。
- `sqlite_session_live`: 验证 SQLite session 的 messages、latest run 和 tool events。
- `builtin_file_tools_live`: 通过 bounded `run_loop()` 验证 `list_files/read_file/grep_search/write_file/edit_file`。
- `file_not_read_recovery_live`: 验证模型收到 `FILE_NOT_READ` 后会自行调用
  `read_file`，再重试 `write_file`。
- `python_tool_live`: 验证 YAML 注册的 Python 自定义工具。
- `permission_path_escape_live`: 验证 `agent.yaml` 中 `permissions.workspace` 能拒绝
  `../` 父目录路径逃逸，底层边界错误包含 `PATH_OUTSIDE_WORKSPACE`。
- `tool_errors_live`: 通过 bounded `run_loop()` 验证 `PERMISSION_ERROR`、`FILE_NOT_READ`、
  `MAX_STEPS_REACHED`、`TOOL_NOT_ALLOWED`。

## 运行命令

全量真实 API 验证：

```bash
uv run python scripts/deepseek_agent_flow.py \
  --work-dir /private/tmp/iris-deepseek-flow-live
```

单场景排障：

```bash
uv run python scripts/deepseek_agent_flow.py \
  --scenario runtime_read_loop_live \
  --work-dir /private/tmp/iris-deepseek-flow-live
```

pytest 中的真实 API gate 默认跳过；显式开启时会通过脚本入口跑 live 场景：

```bash
uv run pytest tests/scripts/test_deepseek_agent_flow.py \
  -m live_deepseek --run-live-deepseek -q
```

运行完成后，`work_dir` 下会保留：

- `summary.md`: 人工快速阅读的摘要报告，包含总体结论、阻塞场景、失败摘要和场景矩阵。
  模块矩阵会汇总每个业务模块的通过状态、调用次数和失败场景；场景矩阵会展示
  `scenario_dir`，便于直接跳转到场景产物。
- `report.json`: 聚合报告，适合后续对照每个场景的预期、实际输出和证据。
  顶层字段包含 `schema_version`、`metadata`、`scenario_catalog`、`total_api_calls`、
  `total_steps`、`module_coverage`、`failure_summary`、`blocking_scenarios` 和
  `blocking_modules`。
  `metadata.environment` 会记录 Python 版本、可执行文件、平台和 git commit / branch /
  dirty 状态，便于把验证结果对应到当时的代码版本。
  每个场景报告包含 `module`、`runtime_api`、`uses_deepseek` 和 `description`。
  场景内的 `api_calls` 表示真实 provider 调用尝试数；发生 retry 时会累计所有尝试，
  并在 `evidence.attempts` 保留每次尝试摘要。provider smoke 即使网络失败，也会记录为
  结构化场景失败而不是丢失在外层异常中。
  使用 `RecordingRuntimeProvider` 的场景会在 `evidence.request_snapshots` 中记录请求摘要，
  包括消息角色顺序、工具 schema 名称、tool choice 和 tool result 回灌标记。
- `logs/`: 运行日志，包含场景开始、结束、失败摘要和报告写入路径。
- `<scenario-name>/`: 场景隔离目录，包含验证所需的临时 agent、workspace、SQLite 文件或工具模块。

## 维护约束

- 保持 `scripts/deepseek_agent_flow.py` 作为稳定入口，不把场景实现重新塞回入口文件。
- 新增 live 场景时，同时更新 `constants.py` 的 `SCENARIO_NAMES`、`catalog.py` 的
  `SCENARIO_CATALOG` 和 `runner.py` 的 `SCENARIO_RUNNERS`。
- 文件写入只能发生在 `--work-dir` 或临时目录下。
- 错误日志和报告不能输出完整 API key。
- 单测只覆盖脚本纯函数、报告、参数、缺 key、recording wrapper 等本地行为；
  真实 API 验证由脚本命令显式运行。
