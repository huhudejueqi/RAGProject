"""OCR 复核资料的轻量识别与 metadata 解析。"""

from __future__ import annotations

import re
from typing import Any

OCR_REVIEWED_CONTENT_TYPE = "ocr_reviewed_text"
OCR_REVIEW_MARKERS = (
    "复核状态：已复核",
    "人工复核：通过",
    "review_status: approved",
    "review_status: reviewed",
    "reviewed: true",
)


def is_reviewed_ocr_text(text: str) -> bool:
    """判断文本是否是已人工复核的 OCR Markdown。

    调用顺序：入库脚本或索引服务 -> is_reviewed_ocr_text()。
    """
    return any(marker in text for marker in OCR_REVIEW_MARKERS)


def parse_ocr_review_metadata(text: str) -> dict[str, Any]:
    """从 OCR Markdown 中提取可复核的轻量 metadata。

    调用顺序：入库脚本或索引服务 -> parse_ocr_review_metadata()。
    """
    if not is_reviewed_ocr_text(text):
        return {}

    metadata: dict[str, Any] = {
        "content_type": OCR_REVIEWED_CONTENT_TYPE,
        "review_status": "reviewed",
    }
    confidence_match = re.search(r"OCR 平均置信度[：:]\s*([0-9.]+)", text)
    if confidence_match:
        metadata["ocr_confidence"] = float(confidence_match.group(1))
    source_match = re.search(r"原始文件[：:]\s*(.+)", text)
    if source_match:
        metadata["ocr_source_path"] = source_match.group(1).strip()
    return metadata
