"""由 Native exec 运行的纯标准库 CSV 转换脚本。"""

import csv
import json
from pathlib import Path


def main() -> None:
    """读取工作目录中的 CSV，生成 JSON。"""
    with Path("input.csv").open(encoding="utf-8", newline="") as source:
        values = [int(row["value"]) for row in csv.DictReader(source)]
    Path("output.json").write_text(
        json.dumps({"rows": len(values), "total": sum(values)}), encoding="utf-8"
    )
    print("output.json ready")


if __name__ == "__main__":
    main()
