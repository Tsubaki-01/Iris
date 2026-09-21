"""从冻结实验数值生成 Memory 选型图表，不执行实验或调用模型。"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ROOT = Path(__file__).resolve().parent
INK = "#243343"
MUTED = "#657487"
GRID = "#E7EBEF"
TEAL = "#43877D"
PURPLE = "#7256A8"
BASE = "#A7B4C2"


def configure() -> None:
    """设置中文字体和统一样式；本机使用 Windows 微软雅黑。"""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
            "svg.fonttype": "path",
        }
    )


def save(fig: Figure, name: str) -> None:
    """保存可直接阅读的 PNG 和适合放大的 SVG。"""
    for extension in ("png", "svg"):
        fig.savefig(ROOT / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def budget_chart(data: dict[str, Any]) -> None:
    """展示相同查询在不同位置下的召回变化。"""
    values = np.array(
        [[np.nan if value is None else value for value in row["values"]] for row in data["rows"]]
    )
    cmap = LinearSegmentedColormap.from_list("memory_hit", ["#F2F5F6", "#92BEB4", TEAL])
    cmap.set_bad("#F1F1F1")
    fig, ax = plt.subplots(figsize=(12.6, 5.4))
    fig.subplots_adjust(left=0.23, right=0.92, top=0.75, bottom=0.16)
    plot = ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    ax.set_xticks(range(4), data["columns"])
    ax.set_yticks(range(len(data["rows"])), [row["label"] for row in data["rows"]])
    ax.tick_params(length=0, pad=12)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "未测" if np.isnan(value) else f"{value:.2f}%"
            ax.text(
                column, row, label, ha="center", va="center", color="white" if value >= 65 else INK
            )
    colorbar = fig.colorbar(plot, ax=ax, fraction=0.025, pad=0.03)
    colorbar.set_label("Hit@5（%，越高越好）", fontsize=10)
    colorbar.outline.set_visible(False)
    fig.text(0.04, 0.94, "查询预算与问题位置对有用命中率的影响", fontsize=18, weight="bold")
    fig.text(0.04, 0.87, "离线检索 · 两份 3,000 条合成语料 · 多样背景条件，k = 5", color=MUTED)
    fig.text(
        0.04,
        0.035,
        "首/中/尾各 144 个正例评估实例；25%/75% 合计 288。变体与语料复本复用相同检索意图。",
        fontsize=10,
        color=MUTED,
    )
    save(fig, "query-budget")


def style_bars(ax: Axes, limit: float, title: str) -> None:
    """为横向条形图统一坐标和网格。"""
    ax.set_xlim(0, limit)
    ax.set_title(title, loc="left", fontsize=12, pad=18, weight="bold")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.invert_yaxis()


def retrieval_chart(data: dict[str, Any]) -> None:
    """把有用命中与无答案噪声分别画出，保留原分母。"""
    rows = data["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.8))
    fig.subplots_adjust(left=0.30, right=0.96, top=0.83, bottom=0.095, hspace=0.43, wspace=0.24)
    for suite_index, (suite, name) in enumerate(
        (("legacy", "基础测试集"), ("supplement", "补充测试集"))
    ):
        for metric in range(2):
            ax = axes[suite_index, metric]
            denominator = data["denominators"][suite][metric]
            values = [row[suite][metric] / denominator * 100 for row in rows]
            metric_name = "有用命中 ↑" if metric == 0 else "无有用记忆题仍返回候选 ↓"
            style_bars(ax, 114, f"{name} · {metric_name}")
            ax.barh(range(len(rows)), values, color=TEAL if metric == 0 else "#B08A74", height=0.58)
            ax.set_yticks(
                range(len(rows)),
                [row["label"] for row in rows] if metric == 0 else [""] * len(rows),
            )
            ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
            for index, row in enumerate(rows):
                ax.text(
                    values[index] + 2,
                    index,
                    f"{row[suite][metric]}/{denominator}",
                    va="center",
                    fontsize=10,
                )
    fig.text(0.04, 0.955, "检索方法对比：有用命中率与候选噪声", fontsize=18, weight="bold")
    fig.text(
        0.04,
        0.907,
        "离线测试集 · 过滤阈值在开发集确定 · 两个题集按各自样本数统计",
        color=MUTED,
    )
    fig.text(
        0.04,
        0.027,
        "条形表示比例，末端标注原始计数。无有用记忆题非空率反映候选噪声，不等同于最终回答错误率。",
        fontsize=10,
        color=MUTED,
    )
    save(fig, "retrieval-options")


def agent_chart(rows: list[dict[str, Any]], group: str) -> None:
    """用同题质量和在线成本呈现完整方案的取舍。"""
    development = group == "development"
    denominator = rows[0]["tasks"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.4 if development else 4.8))
    fig.subplots_adjust(
        left=0.065, right=0.96, top=0.75 if development else 0.68, bottom=0.18, wspace=0.29
    )
    colors = [
        PURPLE if row["label"] == "G" else TEAL if row["label"] == "F" else BASE for row in rows
    ]
    metrics = [
        ("strict_successes", f"完整任务成功 ↑（/ {denominator}）", denominator * 1.14),
        ("score_0_5", "任务宏平均分 ↑（0–5）", 5.65),
        ("online_tokens_per_task", "在线 tokens / 任务 ↓", 98000 if development else 9800),
    ]
    for ax, (field, title, limit) in zip(axes, metrics, strict=True):
        values = [row[field] for row in rows]
        style_bars(ax, limit, title)
        ax.barh(range(len(rows)), values, color=colors, height=0.58)
        ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
        for index, value in enumerate(values):
            if field == "strict_successes":
                label = f"{value}/{denominator}"
            elif field == "score_0_5":
                label = str(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
            else:
                label = f"{value:,.0f}"
            ax.text(
                value + limit * 0.023,
                index,
                label,
                va="center",
                fontsize=10,
                weight="bold" if rows[index]["label"] == "G" else "normal",
            )
        if field == "online_tokens_per_task":
            ticks = [0, 20000, 40000, 60000, 80000] if development else [0, 2000, 4000, 6000, 8000]
            ax.set_xticks(ticks, ["0" if value == 0 else f"{value // 1000}k" for value in ticks])
        elif field == "score_0_5":
            ax.set_xticks(range(6))
        else:
            ax.set_xticks([0, denominator // 3, denominator * 2 // 3, denominator])
    title = "开发集：任务质量与在线成本" if development else "留出集：任务质量与在线成本"
    subtitle = (
        "80/400 条记忆 · 12 场景 × 3 次 = 每组 36 任务 / 60 轮"
        if development
        else "声学领域 80 条记忆 · 12 场景 × 2 次 = 每组 24 任务 / 36 轮"
    )
    fig.text(0.035, 0.94, title, fontsize=18, weight="bold")
    fig.text(0.035, 0.87 if development else 0.835, subtitle, color=MUTED)
    fig.text(
        0.955,
        0.87 if development else 0.835,
        "G：紫色     F：绿色",
        ha="right",
        fontsize=10,
        color=MUTED,
    )
    note = (
        "A/B/C/D 复用前序结果；D0/E/F 同批交错执行，G 分批执行。"
        "在线成本不含概览生成；质量为模型评分。"
        if development
        else "错误断言项零分轮 /36：D0=1，E=1，F=1，G=0。G 沿用同一留出集。"
    )
    fig.text(0.035, 0.048, note, fontsize=10, color=MUTED)
    save(fig, f"agent-{group}")


def main() -> None:
    """读取归档摘录并重绘四组图，不访问原始 API。"""
    data = json.loads((ROOT / "chart-data.json").read_text(encoding="utf-8"))
    configure()
    budget_chart(data["query_budget"])
    retrieval_chart(data["retrieval"])
    agent_chart(data["agent_comparison"]["development"], "development")
    agent_chart(data["agent_comparison"]["holdout"], "holdout")


if __name__ == "__main__":
    main()
