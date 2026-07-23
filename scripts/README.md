# Scripts

这个目录存放 Iris 的本地验证脚本。脚本用于开发期 smoke / integration 验证，不是
运行时 SDK 的一部分。

## deepseek_agent_flow.py

`deepseek_agent_flow.py` 是 DeepSeek live integration 验证入口。它会通过 Iris
集中配置读取 `.env.local` / `.env` 中的 `IRIS_PROVIDER_API_KEYS__DEEPSEEK`
或 `IRIS_API_KEY`，然后运行真实 DeepSeek API 场景。

常用命令：

```bash
uv run python scripts/deepseek_agent_flow.py \
  --work-dir /private/tmp/iris-deepseek-flow-live
```

单场景排障：

```bash
uv run python scripts/deepseek_agent_flow.py \
  --scenario builtin_file_tools_live \
  --work-dir /private/tmp/iris-deepseek-flow-live
```

参数：

- `--work-dir`: 保留验证文件的运行目录；不传则使用临时目录。
- `--scenario`: 选择单个 live 场景；默认 `all`。
- `--retries`: 模型未按要求发起工具调用时的重试次数；默认 `2`。

运行产物：

- `summary.md`: 人工快速阅读的摘要报告，包含总体结论、阻塞场景、失败摘要和场景矩阵。
- `report.json`: 聚合验证报告，包含每个场景的 `expected`、`actual`、`evidence`、
  `error_code` 和 `error_message`，并在顶层记录 `metadata`、`total_api_calls`、
  `total_steps`、`scenario_catalog`、`module_coverage`、`failure_summary`、
  `blocking_scenarios` 和 `blocking_modules`。
  每个场景会记录 `module`、`runtime_api` 和 `scenario_dir`，用于按业务模块定位
  对应 fixture、workspace、SQLite 文件和日志线索。发生 retry 时，`api_calls` 和
  `steps` 记录所有尝试的累计值，`evidence.attempts` 保留每次尝试摘要。
  `metadata.environment` 会记录 Python 版本、可执行文件、平台和 git commit / branch /
  dirty 状态，方便复盘验证对应的代码版本。
- `logs/`: Iris / DeepSeek flow 运行日志。
- `<scenario-name>/`: 每个场景独立的临时 agent、workspace、SQLite 文件或工具 fixture。

实现拆在 `scripts/deepseek_flow/` 下。入口脚本只保留 CLI 兼容层和公开 re-export，
避免单文件继续膨胀。

除 provider 直连 smoke 和 `run_turn` API 语义专项检查外，DeepSeek flow 的业务链路
验证都通过真实 `run_loop()` 执行。

## mock_agent_flow.py

`mock_agent_flow.py` 是本地 mock flow 验证脚本，用于不依赖真实外部 API 的开发期检查。
它和 DeepSeek live flow 的目标不同：mock flow 验证本地装配路径，DeepSeek flow 验证
真实 provider / tool calling / runtime wire format。

## 验证

脚本相关单测：

```bash
uv run pytest tests/scripts/test_deepseek_agent_flow.py -q
```

真实 API pytest gate 默认跳过；需要显式开启：

```bash
uv run pytest tests/scripts/test_deepseek_agent_flow.py \
  -m live_deepseek --run-live-deepseek -q
```

脚本 lint / format：

```bash
uv run ruff check scripts/deepseek_agent_flow.py scripts/deepseek_flow tests/scripts/test_deepseek_agent_flow.py
uv run ruff format --check scripts/deepseek_agent_flow.py scripts/deepseek_flow tests/scripts/test_deepseek_agent_flow.py
```
