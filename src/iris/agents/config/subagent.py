"""Sub Agent catalog 的 raw 解析与只读路由投影。"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Annotated

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ...exceptions import IrisConfigError
from ...tools.subagent import SubagentRoute, SubagentRouteTable


class SubagentCatalogEntry(BaseModel):
    """Catalog entry 的路径与描述边界。"""

    path: Path
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("path")
    @classmethod
    def _resolve_path(cls, value: Path, info: ValidationInfo) -> Path:
        """解析 catalog-relative 路径，不读取 child YAML。"""
        catalog_path: Path = info.context["catalog_path"]
        return (catalog_path.parent / value).resolve()


class SubagentCatalog(BaseModel):
    """完整 catalog 的 schema 与 default membership owner。"""

    default: str
    agents: dict[Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")], SubagentCatalogEntry]

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_default(self) -> SubagentCatalog:
        """Default 必须命中 entry，同时排除空 catalog。"""
        if self.default not in self.agents:
            raise ValueError("default 必须精确命中 agents 中的 selector")
        return self


def load_subagent_catalog(path: str | Path) -> SubagentRouteTable:
    """一次解析 catalog 并冻结路由，不加载任何 child 配置。

    Args:
        path: Catalog YAML 文件路径。

    Returns:
        已规范化、只读的路由快照。

    Raises:
        IrisConfigError: 文件、编码、YAML 或 catalog 字段无效。
    """
    catalog_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        catalog = SubagentCatalog.model_validate(raw, context={"catalog_path": catalog_path})
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise IrisConfigError(
            "Sub Agent catalog 加载失败", path=str(catalog_path), error=str(exc)
        ) from exc
    return SubagentRouteTable(
        default=catalog.default,
        routes=MappingProxyType(
            {
                selector: SubagentRoute(selector, entry.path, entry.description)
                for selector, entry in catalog.agents.items()
            }
        ),
    )
