"""显式发布一个 workspace 文件的副本，供宿主显示或下载。"""

from typing import ClassVar

from pydantic import BaseModel

from ...exceptions import IrisToolExecutionError
from ...message import TextBlock
from .._io import run_tool_io
from ..artifacts import artifact_store_for
from ..base import ToolCapability, ToolExecutionContext, ToolResult
from .file import FileTool


class PublishArtifactInput(BaseModel):
    """要发布的单个文件路径，使用既有文件工具的输入策略。"""

    file_path: str


class PublishArtifactTool(FileTool[PublishArtifactInput]):
    """读取指定文件并保存发布时副本，不扫描目录或更新文件读状态。"""

    name: ClassVar[str] = "publish_artifact"
    description: ClassVar[str] = (
        "将 workspace 内一个已完成写入的文件复制为交付产物，供宿主展示或下载。"
        "显式指定 file_path；源文件后续修改或删除不影响本次副本。"
        "不上传云端，不自动读取图片内容，不替代编辑前的 read_file。"
    )
    input_type: type[PublishArtifactInput] = PublishArtifactInput
    capabilities: ClassVar[set[ToolCapability]] = {ToolCapability.READ}

    async def _impl(
        self, params: PublishArtifactInput, context: ToolExecutionContext
    ) -> ToolResult:
        """在线程内完成文件读取与副本保存，再回到既有工具结果流程。"""

        def publish() -> ToolResult:
            """消费已解析目录边界，只在文件完整复制后返回成功引用。"""
            source = self.file_service.resolve_path(params.file_path, context, write=False)
            if not source.is_file():
                raise IrisToolExecutionError("FILE_NOT_FOUND: 发布路径不是已有文件")
            preview = f"已发布：{source.name}"
            artifact = artifact_store_for(
                context, preview_chars=self.definition.preview_chars
            ).persist_file(context.call_id, source, preview=preview)
            return ToolResult(
                tool_use_id=context.call_id,
                tool_name=self.name,
                content=[
                    TextBlock(
                        text=f"{preview}（{artifact.mime_type}，{artifact.size_bytes} bytes）"
                    )
                ],
                artifact=artifact,
            )

        return await run_tool_io(publish)


__all__ = ["PublishArtifactInput", "PublishArtifactTool"]
