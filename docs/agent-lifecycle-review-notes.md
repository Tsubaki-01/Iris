# Agent lifecycle review notes

本文件记录 lifecycle 各 phase review 中确认不阻塞提交、且正常执行路径几乎不可达的细节项。

## Phase 3

- `Minor`：`StoreRuntimeCommitPort._prepared_record()` 只验证 `run_id`，未再次验证
  `RuntimeToolCall.activation_id` 等于 port 绑定 activation。当前唯一生产调用方
  `AgentRuntime` 总是从同一个 `RuntimeActivationInput` 构造这些 facts，store 还会校验
  command activation fence，因此正常执行路径不可达。若未来允许第三方 engine 实现
  `RuntimeCommitPort` 协议，再补统一的跨 activation prepared-fact 拒绝测试与校验。
