"""FAQ CSV 基础质量检测。"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any


def _resolve_csv_source(row: dict) -> str:
    """从 FAQ CSV 行中解析 source 字段。

    调用顺序：质量门禁流程 -> _resolve_csv_source()。
    """
    # 兼容中英文列名：优先取英文 "source"，再尝试中文 "业务分类"、"分类" 等常见列名
    # 这样同一份 CSV 处理代码可以同时支持国内团队和国际化团队的导出格式
    return str(
        row.get("source") or row.get("source_filter") or row.get("业务分类")
        or row.get("分类") or row.get("subject_name") or ""
    ).strip()


def read_faq_records(csv_path: str | Path) -> list[dict[str, Any]]:
    """读取 FAQ 记录，兼容中文和英文列名。返回行号、问题、答案和 source。

    调用顺序：质量门禁流程 -> read_faq_records()。
    """
    path = Path(csv_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        # row_index 从 2 开始（1 是表头），便于向用户展示 CSV 中的实际行号
        for row_index, row in enumerate(reader, start=2):
            records.append(
                {
                    "row": row_index,
                    # 兼容中英文列名："问题"/"question"、"答案"/"answer"
                    "question": str(row.get("问题") or row.get("question") or "").strip(),
                    "answer": str(row.get("答案") or row.get("answer") or "").strip(),
                    "source": _resolve_csv_source(row),
                }
            )
    return records


def analyze_faq_csv(csv_path: str | Path, valid_sources: list[str]) -> dict[str, Any]:
    """检查 FAQ CSV 的基础质量（空答案、重复问题、分类不在白名单等）。

    调用顺序：质量门禁流程 -> analyze_faq_csv()。
    """
    path = Path(csv_path)
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "record_count": 0,
        "valid_record_count": 0,
        "empty_question_rows": [],
        "empty_answer_rows": [],
        "duplicate_questions": [],
        "invalid_sources": [],
        "source_counts": {},
    }
    # FAQ CSV 不存在时直接返回错误信息而非抛异常，避免质量报告整体失败
    if not path.exists():
        result["error"] = "FAQ CSV 不存在。"
        return result

    question_seen: dict[str, int] = {}
    source_counter: Counter[str] = Counter()
    for record in read_faq_records(path):
        question = record["question"]
        answer = record["answer"]
        source = record["source"]
        row_index = record["row"]
        result["record_count"] += 1
        # 空问题/空答案行分别记录行号但不中断，让用户一次性看到所有质量问题
        if not question:
            result["empty_question_rows"].append(row_index)
        if not answer:
            result["empty_answer_rows"].append(row_index)
        if question:
            question_seen[question] = question_seen.get(question, 0) + 1
        if source:
            source_counter[source] += 1
            # source 不在场景白名单中时记录，用于发现 FAQ 分类配置错误
            if source not in valid_sources:
                result["invalid_sources"].append({"row": row_index, "source": source})
        # 同时有 question 和 answer 才计为有效记录
        if question and answer:
            result["valid_record_count"] += 1
    # 筛选出出现次数 > 1 的问题作为重复问题
    result["duplicate_questions"] = [question for question, count in question_seen.items() if count > 1]
    result["source_counts"] = dict(source_counter)
    return result
