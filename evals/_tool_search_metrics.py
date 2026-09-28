"""工具选择实验共用的命中与误选统计。"""

from typing import Any


def selection_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    """按同一口径统计正例命中和负例非空选择，空分母记为 null。"""
    positive = [row for row in rows if row["expected"]]
    negative = [row for row in rows if not row["expected"]]
    result: dict[str, Any] = {"positive": len(positive), "negative": len(negative)}
    result["candidate_recall"] = (
        sum(bool(set(row["expected"]) & set(row["candidates"])) for row in positive) / len(positive)
        if positive
        else None
    )
    for k in (1, 3):
        result[f"top{k}"] = (
            sum(bool(set(row["expected"]) & set(row[field][:k])) for row in positive)
            / len(positive)
            if positive
            else None
        )
    result["false_selection"] = (
        sum(bool(row[field]) for row in negative) / len(negative) if negative else None
    )
    return result
