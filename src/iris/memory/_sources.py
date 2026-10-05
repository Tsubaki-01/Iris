"""从既有经历和事件关系投影自动维护来源；不另存运行状态。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

from .generation_models import MemorySource

# 同一投影用于选材、重选、计数与提交批次来源，资格过滤始终先于 LIMIT。
SOURCE_CTE = """
WITH episode_sources AS (
    SELECT e.namespace,e.id,
        json_extract(e.payload,'$.metadata.lifecycle_source_id') AS source_id,
        json_extract(e.payload,'$.source_id') AS run_id,
        json_extract(e.payload,'$.metadata.session_id') AS session_id,
        (s.terminal_message_count IS NOT NULL
            AND s.captured_until >= s.terminal_message_count) AS complete
    FROM memory_episodes e LEFT JOIN memory_capture_sources s
        ON s.namespace=e.namespace
        AND s.lifecycle_source_id=json_extract(e.payload,'$.metadata.lifecycle_source_id')
        AND s.run_id=json_extract(e.payload,'$.source_id')
    WHERE json_extract(e.payload,'$.metadata.lifecycle_source_id') IS NOT NULL
), episode_calls AS (
    SELECT e.namespace,e.id,json_array(
        json_extract(e.payload,'$.metadata.lifecycle_source_id'),
        json_extract(e.payload,'$.source_id'),
        json_extract(r.value,'$.metadata.call_id')
    ) AS call_source_id
    FROM memory_episodes e,json_each(e.payload,'$.records') r
    WHERE json_extract(r.value,'$.metadata.call_id') IS NOT NULL
), event_sources AS (
    SELECT h.namespace,h.id,s.source_id,s.run_id,s.session_id,coalesce(s.complete,0) AS complete
    FROM memory_events h LEFT JOIN episode_calls c
        ON c.namespace=h.namespace AND c.call_source_id=json_extract(h.payload,'$.source_id')
    LEFT JOIN episode_sources s ON s.namespace=c.namespace AND s.id=c.id
    WHERE json_extract(h.payload,'$.source_type')='tool_event'
), input_sources AS (
    SELECT namespace,'episode' AS kind,id,source_id,run_id,session_id,complete
    FROM episode_sources
    UNION
    SELECT o.namespace,'observation',o.id,s.source_id,s.run_id,s.session_id,s.complete
    FROM memory_observations o,json_each(o.payload,'$.evidence') r
    JOIN episode_sources s ON s.namespace=o.namespace
        AND s.id=json_extract(r.value,'$.source_id')
    WHERE json_extract(r.value,'$.kind')='episode'
    UNION
    SELECT o.namespace,'observation',o.id,s.source_id,s.run_id,s.session_id,s.complete
    FROM memory_observations o,json_each(o.payload,'$.evidence') r
    JOIN event_sources s ON s.namespace=o.namespace
        AND s.id=json_extract(r.value,'$.source_id')
    WHERE json_extract(r.value,'$.kind')='event'
    UNION
    SELECT c.namespace,'change',c.event_id,s.source_id,s.run_id,s.session_id,s.complete
    FROM memory_dream_changes c JOIN event_sources s
        ON s.namespace=c.namespace AND s.id=c.event_id
)
"""


def source_filter(
    kind: str, alias: str, id_column: str, allowed_sources: frozenset[tuple[str, str]] | None
) -> tuple[str, list[str]]:
    """生成当前查询的来源限制；None 保留显式 SDK 选材语义。"""
    if allowed_sources is None:
        return "", []
    return (
        f" AND NOT EXISTS (SELECT 1 FROM input_sources src WHERE src.kind='{kind}'"
        f" AND src.namespace={alias}.namespace AND src.id={alias}.{id_column}"
        " AND (src.complete=0 OR NOT EXISTS (SELECT 1 FROM json_each(?) allowed"
        " WHERE json_extract(allowed.value,'$[0]')=src.source_id"
        " AND json_extract(allowed.value,'$[1]')=src.run_id)))",
        [json.dumps(sorted(allowed_sources))],
    )


def read_sources(
    connection: sqlite3.Connection,
    namespace: str,
    selections: Sequence[tuple[str, Sequence[str]]],
) -> tuple[MemorySource, ...]:
    """读取一个固定消费批次的来源，不包含比较用的已发布知识。"""
    clauses = []
    params: list[str] = [namespace]
    for kind, ids in selections:
        clauses.append("(kind=? AND id IN (SELECT value FROM json_each(?)))")
        params.extend((kind, json.dumps(list(ids))))
    rows = connection.execute(
        SOURCE_CTE + " SELECT DISTINCT source_id,run_id,session_id FROM input_sources"
        " WHERE namespace=? AND source_id IS NOT NULL AND ("
        + " OR ".join(clauses)
        + ") ORDER BY source_id,run_id,session_id",
        params,
    )
    return tuple(MemorySource(*row) for row in rows)
