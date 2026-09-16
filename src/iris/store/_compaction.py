"""两种 store 共用的摘要投影增量检查与独立用量累加。"""

from ..exceptions import IrisRunConflictError, IrisRunStateError
from ..lifecycle.models import (
    CheckpointResumability,
    RunCheckpoint,
    RunRecord,
    RunUsage,
    SessionCompaction,
    TokenUsage,
)
from ..lifecycle.store import CommitCompaction


def add_compaction_usage(usage: RunUsage, increment: TokenUsage) -> RunUsage:
    """在可信用量对象中只累计摘要调用三项 token 数。"""
    current = usage.compaction
    return usage.model_copy(
        update={
            "compaction": TokenUsage.model_construct(
                input_tokens=current.input_tokens + increment.input_tokens,
                output_tokens=current.output_tokens + increment.output_tokens,
                total_tokens=current.total_tokens + increment.total_tokens,
            ),
        }
    )


def validate_compaction_commit(
    run: RunRecord,
    current_checkpoint: RunCheckpoint,
    command: CommitCompaction,
    *,
    session_message_count: int,
    previous_compaction: SessionCompaction | None,
) -> None:
    """检查投影提交影响的执行位置、覆盖范围和取消事实。"""
    if run.cancellation_requested_at is not None:
        raise IrisRunStateError("已取消 run 不能提交摘要投影", run_id=run.run_id)
    if (
        current_checkpoint.resumability is not CheckpointResumability.SAFE
        or current_checkpoint.engine_cursor.get("position") != "before_model"
    ):
        raise IrisRunStateError("摘要投影要求 SAFE before_model checkpoint")
    if run.usage.model_steps_reserved != run.usage.model_steps_committed + 1:
        raise IrisRunStateError("摘要投影要求一个 pending model-step reservation")
    previous_count = 0 if previous_compaction is None else previous_compaction.covered_message_count
    if not previous_count < command.compaction.covered_message_count <= session_message_count:
        raise IrisRunStateError("摘要覆盖必须前移且不能超过原文消息数")
    if command.checkpoint.engine_cursor != current_checkpoint.engine_cursor:
        raise IrisRunConflictError("摘要投影不能改变 engine cursor")
    if command.checkpoint.resumability is not current_checkpoint.resumability:
        raise IrisRunConflictError("摘要投影不能改变 checkpoint resumability")
