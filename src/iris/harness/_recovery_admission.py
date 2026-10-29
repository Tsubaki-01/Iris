"""共享恢复 owner 的执行准入包装。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine
from functools import wraps
from typing import TYPE_CHECKING, Any, Concatenate, Protocol

if TYPE_CHECKING:
    from .recovery_point import RootRecoveryCoordinator


class RecoveryParticipant(Protocol):
    """Runner、Manager 与 Goal 控制都借用同一个 root owner。"""

    @property
    def _recovery(self) -> RootRecoveryCoordinator | None: ...


def recovery_operation[Owner: RecoveryParticipant, Result, **Parameters](
    operation: Callable[Concatenate[Owner, Parameters], Awaitable[Result]],
) -> Callable[Concatenate[Owner, Parameters], Coroutine[Any, Any, Result]]:
    """在首次 await 前登记普通操作，关闭能力时直接调用原路径。"""

    @wraps(operation)
    async def invoke(self: Owner, *args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
        if self._recovery is None:
            return await operation(self, *args, **kwargs)
        with self._recovery.operation():
            return await operation(self, *args, **kwargs)

    return invoke


def recovery_mutation[Owner: RecoveryParticipant, Result, **Parameters](
    operation: Callable[Concatenate[Owner, Parameters], Result],
) -> Callable[Concatenate[Owner, Parameters], Result]:
    """同步 mutation 在任何写入之前取得准入。"""

    @wraps(operation)
    def invoke(self: Owner, *args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
        if self._recovery is None:
            return operation(self, *args, **kwargs)
        with self._recovery.operation():
            return operation(self, *args, **kwargs)

    return invoke


def recovery_internal_operation[Owner: RecoveryParticipant, Result, **Parameters](
    operation: Callable[Concatenate[Owner, Parameters], Awaitable[Result]],
) -> Callable[Concatenate[Owner, Parameters], Coroutine[Any, Any, Result]]:
    """内部调度在取得自己的锁前等待快照释放。"""

    @wraps(operation)
    async def invoke(self: Owner, *args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
        if self._recovery is None:
            return await operation(self, *args, **kwargs)
        async with self._recovery.internal_operation():
            return await operation(self, *args, **kwargs)

    return invoke
