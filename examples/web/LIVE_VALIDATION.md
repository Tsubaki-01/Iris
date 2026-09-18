# Web 真实 API 验证记录

验证日期：2026-09-18。凭据通过项目现有 `init_config(env_file=".env")` 加载。
Tavily Search / Extract 与 DeepSeek 均使用真实 API，未替换 HTTP transport 或聊天 provider。
请求和输出证据不保存认证头或配置中的密钥。

## 结果

16 个真实场景经过首轮与针对性复验全部通过；相关离线回归为 **69 passed，16 live skipped**。
Ruff 检查与格式检查通过。新增可运行示例、真实测试与使用说明；后续补充修复了正式 CLI
入口的 UTF-8 输出，Web 工具内核未改动。
正式 CLI 修复后的精准回归（`tests/cli` 与 `tests/examples/test_web_examples.py`）为 **32 passed**。

| 场景 | 实际结果 |
| --- | --- |
| 默认搜索 | 返回带标题、URL 和片段的来源，数量不超过默认 10 |
| 包含 + 排除域名、max_results=3 | 返回来源全部属于 docs.python.org，排除 peps.python.org 生效 |
| 单独排除域名 | 返回来源不包含 docs.python.org |
| day / week / month / year | 四种参数均被服务接受，正常返回，数量不超过 3 |
| 合法后缀但不存在的限定子域名 | 返回 Results: 0，工具成功 |
| 服务拒绝 .invalid 顶级域名 | 真实 HTTP 400 转换为 EXECUTION_ERROR，保留服务原因 |
| 两 URL 全文 | 成功 2、失败 0，完整 Markdown 共 117,272 字符 |
| 长结果 artifact 与续读 | 默认阈值触发落盘；3 次 read_file 拼接后与完整正文逐字符一致 |
| 两 URL 的 TaskGroup 摘录 | 成功 2，输出 3,667 字符，正文含取消任务及 ExceptionGroup 信息 |
| 同页换为 to_thread query | 输出 2,323 字符，包含 to_thread 和 blocking 内容，摘录随问题变化 |
| 部分失败，全文 / 摘录 | 两种模式均保留成功正文，并在前面列出失败 URL 与原因 |
| 全部失败，全文 / 摘录 | 两种模式均返回 EXECUTION_ERROR，保留失败 URL 与原因 |
| 真实 Agent 与 SQLite | 3 个模型步骤、2 个工具调用均 committed，无工具错误，形成引用回答；结果与会话工具记录可从新 SQLiteStore 重读 |

最终 Agent 用例的 run_id 为 `run_2a21d97d0f684f36b84fc39668aff130`，记录的 token 用量为
输入 6,369、输出 901、总计 7,270。模型先搜索，然后读取
`https://docs.python.org/3/library/asyncio-task.html` 的相关摘录，最后回答并引用该来源。

另直接执行了两个示例命令：独立搜索退出码 0，返回 1 条 Python 文档来源；Agent 命令退出码 0，
`terminal / completed`，3 个模型步骤，回答 2,082 字符，默认 SQLite 路径正常写入。
这次命令运行的 run_id 为 `run_751457eda65347a8bf07d6ac3cb78a08`。

## 验证中修正的事项

- 初始空结果样本使用 `.invalid`，实际触发 Tavily HTTP 400。保留它作为真实 HTTP 错误测试，
  并使用不存在的 `iris-web-example-missing.python.org` 验证真正的零结果。
- 初始 Agent 断言要求 URL 字符串完全相同。模型省略了 Python 文档的 `highlight=cancel`，
  但域名、版本、语言和文档路径一致。测试按该文档语义允许省略高亮参数和锚点，其余查询参数
  仍参与比较；这不是 Iris 抓取失败，也不构成模型始终遵循 URL 原样复制的保证。
- 真实 CLI 输出包含 `¶`，Windows 默认 GBK 管道触发 `UnicodeEncodeError`。先新增离线复现用例，
  再将两个示例命令入口设为 UTF-8。回归与真实命令复验均通过。
  随后确认正式 `iris chat` 的默认输出同样受影响，已在 `src/iris/cli/main.py` 的入口统一设置
  UTF-8；新增回归覆盖 GBK 管道下中文、`¶` 和 emoji 的完整输出。

## 边界与观察

- 时间范围只验证真实服务接受参数及输出结构；当前工具不返回可靠发布日期，不能证明每条结果
  都严格处于指定日期内。
- 全文和 query 摘录验证了本次页面与问题的返回内容，不保证每个网页都能提取，也不承诺摘录固定
  长度。Iris 保留服务内容，长结果由已有 artifact 机制处理。
- 限流、超时、非法响应等分支由现有离线测试覆盖，没有通过耗尽额度或制造外部故障复现。
- 真实模型运行结束时，LiteLLM 输出 `Logging.async_success_handler was never awaited` 警告。
  本次运行状态、工具结果和持久化断言均通过；本轮未修改其日志协程生命周期。
- 实时索引、页面和模型决策会变化；本报告记录本次执行，不保证未来运行结果逐字一致。

接口参考：[Tavily Search](https://docs.tavily.com/documentation/api-reference/endpoint/search)、
[Tavily Extract](https://docs.tavily.com/documentation/api-reference/endpoint/extract)。

## 本地证据

以下文件属于当前工作区的 `tmp/` 或 `.iris/`，被 Git 忽略；复现方法见 [README](README.md#验证与复现)。

- [首轮结果：13 passed、2 failed](../../tmp/web-live-20260918.xml)，
  [原始工具与 Agent 证据](../../tmp/pytest-web-live-20260918/)。
- [空结果与 HTTP 错误复验：2 passed、1 failed](../../tmp/web-live-followup-20260918.xml)。
- [最终 Agent 复验：1 passed](../../tmp/web-agent-final-20260918.xml)，
  [RunResult](../../tmp/pytest-web-agent-final-20260918/test_real_model_search_fetch_a0/run-result.json)，
  [工具调用](../../tmp/pytest-web-agent-final-20260918/test_real_model_search_fetch_a0/tool-calls.json)，
  [SQLite](../../tmp/pytest-web-agent-final-20260918/test_real_model_search_fetch_a0/.iris/web.db)。
- [最终离线回归](../../tmp/web-offline-final-20260918.xml)。
- [独立搜索命令输出](../../tmp/web-search-cli-20260918.txt)、
  [Agent 命令输出](../../tmp/web-agent-cli-20260918.json)、[示例 SQLite](.iris/web.db)。
