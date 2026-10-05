"""自进化取得的领域说明与实际生成请求使用同一输出契约。"""

from pathlib import Path

import pytest

from iris.memory import MemoryObserveInput, generation_prompt_descriptions

from .test_generation import Provider, service


@pytest.mark.asyncio
async def test_flush_description_matches_actual_request_contract(tmp_path: Path) -> None:
    """说明不复制响应模型，实际请求仍包含同一份不可编辑协议。"""
    provider = Provider(lambda source: {"observations": []})
    memory = service(tmp_path, provider)
    memory.observe(MemoryObserveInput(text="本项目使用 uv"))
    await memory.flush("project")
    description, variables = generation_prompt_descriptions()["memory_flush"]
    assert variables == {}
    assert provider.requests[0].messages[0].text.endswith(description)
