"""本地 Docker 资源准备与物理生命周期，不拥有命令调用或停止收据。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from ..exceptions import IrisSandboxError
from .config import DockerConfig
from .snapshot import DockerSnapshot

if TYPE_CHECKING:
    from aiodocker import Docker
    from aiodocker.containers import DockerContainer
    from aiohttp import ClientTimeout

CONTROL_SECONDS = 10.0


class DockerSandbox:
    """持有一个 owner 的本地容器资源，由调用方串行协调生命周期。

    Args:
        workspace_root (Path): 已解析的挂载目录。
        config (DockerConfig): 已解析的本地 Docker 配置。
        workspace_writable (bool): 工作区挂载是否可写。
        owner_id (str): 创建容器的唯一 owner 标识。
    """

    def __init__(
        self,
        workspace_root: Path,
        config: DockerConfig,
        *,
        workspace_writable: bool,
        owner_id: str,
        restored_snapshot: DockerSnapshot | None = None,
    ) -> None:
        self._workspace_root = workspace_root
        self._config = config
        self._restored_snapshot = restored_snapshot
        self._image_id = ""
        self._endpoint: str | None = None
        self._workspace_writable = workspace_writable
        self._owner_id = owner_id
        self._container_name = f"iris-command-{owner_id}"
        self._client: Docker | None = None
        self._image_environment: dict[str, str] = {}
        self._docker_error: type[Exception] = OSError
        self._creation_issued = False
        self._needs_stop = False
        self._running = False
        self.container: DockerContainer | None = None
        self.user = "1000:1000"
        self.control_timeout: ClientTimeout | None = None
        self.driver_errors: tuple[type[Exception], ...] = (OSError, TimeoutError)

    async def prepare(self) -> None:
        """准备 driver、Linux engine 与现有镜像，不创建、构建或拉取资源。"""
        try:
            from aiodocker import Docker
            from aiodocker.exceptions import DockerError
            from aiohttp import ClientError, ClientTimeout
        except ImportError as error:
            raise IrisSandboxError("Docker 模式需要安装 Iris 的 sandbox extra") from error
        self.driver_errors = (DockerError, ClientError, OSError, TimeoutError)
        self._docker_error = DockerError
        self.control_timeout = ClientTimeout(total=CONTROL_SECONDS)
        endpoint = self._config.endpoint or (
            "npipe:////./pipe/docker_engine" if os.name == "nt" else "unix:///var/run/docker.sock"
        )
        self._endpoint = endpoint
        try:
            self._client = Docker(url=endpoint, timeout=self.control_timeout)
            async with asyncio.timeout(CONTROL_SECONDS):
                info = await self._client.system.info()
                if info["OSType"] != "linux":
                    raise IrisSandboxError("Docker 模式只支持 Linux containers")
                image = await self._client.images.inspect(self._config.image)
            self._image_id = image["Id"]
            if os.name != "nt" and "docker desktop" not in info["OperatingSystem"].lower():
                self.user = f"{os.getuid()}:{os.getgid()}"
            self._image_environment = dict(
                entry.split("=", 1) for entry in image["Config"].get("Env", [])
            )
            if self._restored_snapshot is not None:
                self.user = self._restored_snapshot.user
        except self.driver_errors as error:
            raise IrisSandboxError(
                "Docker 准备失败；请确认本地引擎可用，并将配置的镜像预先显式构建到该引擎",
                image=self._config.image,
                error=str(error),
            ) from error

    async def create(self) -> None:
        """惰性创建本实例容器，保留响应不确定时按名称回收的事实。"""
        if self.container is not None:
            return
        self._creation_issued = True
        self._needs_stop = True
        try:
            try:
                async with asyncio.timeout(CONTROL_SECONDS):
                    self.container = await cast("Docker", self._client).containers.create(
                        self._container_config(), name=self._container_name
                    )
            except self._docker_error as error:
                if 400 <= error.status < 500:
                    self._creation_issued = False
                    self._needs_stop = False
                raise
        except self.driver_errors as error:
            raise IrisSandboxError("Docker 容器创建失败", error=str(error)) from error

    async def start(self) -> None:
        """启动已创建的容器，已运行时保持原资源。"""
        if self._running:
            return
        self._needs_stop = True
        try:
            async with asyncio.timeout(CONTROL_SECONDS):
                await cast("DockerContainer", self.container).start()
            self._running = True
        except self.driver_errors as error:
            raise IrisSandboxError("Docker 容器启动失败", error=str(error)) from error

    async def stop(self) -> None:
        """停止已知或创建结果不确定的容器，并确认停止事实。"""
        try:
            if self.container is None and self._creation_issued:
                async with asyncio.timeout(CONTROL_SECONDS):
                    self.container = await cast("Docker", self._client).containers.get(
                        self._container_name
                    )
            if self.container is not None and self._needs_stop:
                try:
                    async with asyncio.timeout(CONTROL_SECONDS):
                        await self.container.stop(t=0, timeout=self.control_timeout)
                except self.driver_errors as error:
                    async with asyncio.timeout(CONTROL_SECONDS):
                        state = await self.container.show()
                    if state["State"]["Running"]:
                        raise IrisSandboxError("Docker 容器仍运行，停止未确认") from error
            self._running = False
            self._needs_stop = False
        except self.driver_errors as error:
            raise IrisSandboxError("Docker 容器停止未确认", error=str(error)) from error

    async def aclose(self) -> None:
        """删除调用方已停止的容器并关闭 client，不发起第二次停止。"""
        await self.delete_container()
        await self.close_client()

    async def delete_container(self) -> None:
        """删除本实例已停止的容器，client 留给后续镜像清理。"""
        try:
            if self.container is not None:
                async with asyncio.timeout(CONTROL_SECONDS):
                    await self.container.delete(timeout=self.control_timeout)
                self.container = None
                self._creation_issued = False
        except self.driver_errors as error:
            raise IrisSandboxError("Docker 容器删除未完成", error=str(error)) from error

    async def close_client(self) -> None:
        """关闭已完成资源清理的 client。"""
        try:
            if self._client is not None:
                async with asyncio.timeout(CONTROL_SECONDS):
                    await self._client.close()
                self._client = None
        except self.driver_errors as error:
            raise IrisSandboxError("Docker 资源关闭未完成", error=str(error)) from error

    async def freeze_for_snapshot(self) -> bool:
        """只暂停本次读取到的运行中容器，响应失败时收回实际暂停。"""
        if self.container is None:
            return False
        try:
            async with asyncio.timeout(CONTROL_SECONDS):
                state = await self.container.show()
            if not state["State"]["Running"] or state["State"]["Paused"]:
                return False
            try:
                async with asyncio.timeout(CONTROL_SECONDS):
                    await self.container.pause()
            except self.driver_errors:
                async with asyncio.timeout(CONTROL_SECONDS):
                    state = await self.container.show()
                if state["State"]["Paused"]:
                    await self.release_snapshot(paused_here=True)
                raise
            return True
        except self.driver_errors as error:
            raise IrisSandboxError("Docker 快照暂停失败", error=str(error)) from error

    async def commit_snapshot(self, *, source_id: str, snapshot_id: UUID) -> DockerSnapshot:
        """保存冻结容器，或为尚未创建容器的确切基础镜像增加自有 tag。"""
        from aiohttp import ClientTimeout

        repository = f"iris-recovery/{source_id}"
        tag = str(snapshot_id)
        try:
            if self.container is None:
                config = self._container_config()
                host = config["HostConfig"]
                writable = self._workspace_writable
                image_id = self._image_id
                async with asyncio.timeout(CONTROL_SECONDS):
                    await cast("Docker", self._client).images.tag(image_id, repository, tag=tag)
            else:
                async with asyncio.timeout(CONTROL_SECONDS):
                    state = await self.container.show()
                config = state["Config"]
                host = state["HostConfig"]
                writable = next(
                    mount["RW"] for mount in state["Mounts"] if mount["Destination"] == "/workspace"
                )
                committed = await self.container.commit(
                    repository=repository, tag=tag, pause=False, timeout=ClientTimeout(total=None)
                )
                image_id = committed["Id"]
            return DockerSnapshot.model_construct(
                image_ref=f"{repository}:{tag}",
                config=DockerConfig.model_construct(
                    image=image_id,
                    endpoint=self._endpoint,
                    network=host["NetworkMode"],
                    cpus=host["NanoCpus"] / 1_000_000_000,
                    memory_mb=host["Memory"] // (1024 * 1024),
                    pids_limit=host["PidsLimit"],
                    environment=dict(entry.split("=", 1) for entry in config["Env"]),
                ),
                user=config["User"],
                working_dir=config["WorkingDir"],
                entrypoint=tuple(config["Entrypoint"]),
                command=tuple(config["Cmd"]),
                workspace_writable=writable,
            )
        except self.driver_errors as error:
            raise IrisSandboxError("Docker 快照镜像保存失败", error=str(error)) from error

    async def release_snapshot(self, *, paused_here: bool) -> None:
        """撤销本次暂停，停止或未创建的容器保持原状。"""
        if not paused_here:
            return
        try:
            async with asyncio.timeout(CONTROL_SECONDS):
                await cast("DockerContainer", self.container).unpause()
        except self.driver_errors as error:
            raise IrisSandboxError(
                "Docker 快照解除暂停失败", operation="unpause", error=str(error)
            ) from error

    async def prune_snapshot_images(
        self, *, source_id: str, keep_image_ref: str | None
    ) -> tuple[str, ...]:
        """按本 source 的 tag 清理旧镜像，延期当前容器直接依赖的镜像。"""
        repository = f"iris-recovery/{source_id}"
        deferred = []
        try:
            async with asyncio.timeout(CONTROL_SECONDS):
                active_image = (
                    (await self.container.show())["Image"] if self.container is not None else None
                )
                client = cast("Docker", self._client)
                images = await client.images.list(filters={"reference": [f"{repository}:*"]})
                for image in images:
                    for ref in image["RepoTags"] or ():
                        if not ref.startswith(f"{repository}:") or ref == keep_image_ref:
                            continue
                        if image["Id"] == active_image:
                            deferred.append(ref)
                        else:
                            await client.images.delete(ref, force=False, noprune=True)
            return tuple(sorted(deferred))
        except self.driver_errors as error:
            raise IrisSandboxError("Docker 快照引用清理失败", error=str(error)) from error

    def _container_config(self) -> dict[str, Any]:
        environment = dict(self._image_environment)
        environment.update(
            HOME="/tmp",
            PYTHONUSERBASE="/tmp/.local",
            PATH=f"/tmp/.local/bin:{environment.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
        )
        environment.update(self._config.environment)
        restored = self._restored_snapshot
        if restored is not None:
            environment = restored.config.environment
        return {
            "Image": self._config.image,
            "Entrypoint": ["sleep"] if restored is None else list(restored.entrypoint),
            "Cmd": ["infinity"] if restored is None else list(restored.command),
            "User": self.user,
            "WorkingDir": "/workspace" if restored is None else restored.working_dir,
            "Env": [f"{key}={value}" for key, value in environment.items()],
            "Labels": {"iris.sandbox": "true", "iris.command.owner": self._owner_id},
            "HostConfig": {
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": str(self._workspace_root),
                        "Target": "/workspace",
                        "ReadOnly": not self._workspace_writable,
                    }
                ],
                "Privileged": False,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges:true"],
                "Init": True,
                "AutoRemove": False,
                "RestartPolicy": {"Name": "no"},
                "NetworkMode": self._config.network,
                "NanoCpus": int(self._config.cpus * 1_000_000_000),
                "Memory": self._config.memory_mb * 1024 * 1024,
                "PidsLimit": self._config.pids_limit,
            },
        }


__all__ = ["DockerSandbox"]
