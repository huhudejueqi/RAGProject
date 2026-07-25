"""把人工复核后的 Bad Case 合并为正式回归样本。

这个脚本承接 `extract_bad_cases_from_report.py` 的输出：

1. 先从评测报告里抽出疑似异常样本；
2. 人工补齐/修正 `expected_*` 和 `grading_notes`；
3. 再把复核后的样本合并进正式 `eval_sets/*.json` 回归集。

默认按 `source_case_id` 作为稳定合并键：bad case 里的 `case_id` 会退回原始样本
编号，方便长期回归集保持稳定命名。

用法示例：

    python scripts/promote_bad_cases_to_regression.py \
      --source eval_sets/local_bad_cases.json \
      --target eval_sets/enterprise_it_troubleshooting_cases.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import PROJECT_ROOT, configure_utf8_stdio, print_json


DEFAULT_SOURCE = PROJECT_ROOT / "eval_sets" / "local_bad_cases.json"


def project_path(path: str | Path) -> Path:
    """把命令行路径解析成项目内绝对路径。"""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSON 数组文件。"""

    file_path = project_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"文件不是 JSON 数组：{file_path}")
    return [item for item in payload if isinstance(item, dict)]


def write_json_list(path: str | Path, payload: list[dict[str, Any]]) -> Path:
    """写出 JSON 数组文件。"""

    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def first_non_empty(*values: Any) -> Any:
    """返回第一个非空值。"""

    for value in values:
        if value not in (None, "", [], {}, ()):
            return value
    return None


def copy_if_present(target: dict[str, Any], source: dict[str, Any], keys: tuple[str, ...]) -> None:
    """从 source 复制非空字段到 target。"""

    for key in keys:
        value = source.get(key)
        if value not in (None, "", [], {}, ()):
            target[key] = value


def normalize_promoted_case(item: dict[str, Any]) -> dict[str, Any]:
    """把 bad case 转成正式回归样本。"""

    original_case_id = str(item.get("source_case_id") or item.get("case_id") or "").strip()
    if not original_case_id:
        raise ValueError("bad case 缺少 source_case_id / case_id，无法合并到回归集")

    query = str(first_non_empty(item.get("query"), item.get("question")) or "").strip()
    if not query:
        raise ValueError(f"bad case {original_case_id} 缺少 query / question")

    promoted: dict[str, Any] = {
        "case_id": original_case_id,
        "query": query,
    }

    bad_case_id = str(item.get("case_id") or "").strip()
    if bad_case_id and bad_case_id != original_case_id:
        promoted["bad_case_id"] = bad_case_id

    copy_if_present(
        promoted,
        item,
        (
            "scenario_id",
            "source_filter",
            "tenant_id",
            "dataset_id",
            "visibility",
            "user_role",
            "kb_version",
            "expected_hit_type",
            "expected_effective_source",
            "expected_prompt_profile",
            "expected_source_contains",
            "expected_keywords",
            "grading_notes",
            "bad_case_reasons",
        ),
    )
    return promoted


def merge_cases(
    base_items: list[dict[str, Any]],
    incoming_items: list[dict[str, Any]],
    *,
    conflict: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """把 incoming 样本合并进 base 样本。"""

    if conflict not in {"replace", "skip", "error"}:
        raise ValueError("conflict 只支持 replace / skip / error")

    merged = list(base_items)
    index_by_case_id = {str(item.get("case_id") or ""): idx for idx, item in enumerate(merged) if str(item.get("case_id") or "")}
    stats = {"inserted": 0, "replaced": 0, "skipped": 0}

    for item in incoming_items:
        case_id = str(item.get("case_id") or "").strip()
        if not case_id:
            raise ValueError("待合并样本缺少 case_id")
        if case_id in index_by_case_id:
            if conflict == "skip":
                stats["skipped"] += 1
                continue
            if conflict == "error":
                raise ValueError(f"发现重复 case_id：{case_id}")
            merged[index_by_case_id[case_id]] = item
            stats["replaced"] += 1
        else:
            index_by_case_id[case_id] = len(merged)
            merged.append(item)
            stats["inserted"] += 1
    return merged, stats


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """构建 bad case -> regression 集的合并计划。"""

    source_path = project_path(args.source)
    target_path = project_path(args.target)
    source_items = load_json_list(source_path)
    target_items = load_json_list(target_path) if target_path.exists() else []
    promoted_items = [normalize_promoted_case(item) for item in source_items]
    merged_items, stats = merge_cases(target_items, promoted_items, conflict=args.conflict)

    output_path = project_path(args.output) if args.output else target_path
    plan = {
        "report_type": "bad_case_regression_promotion_plan",
        "ok": True,
        "source": str(source_path),
        "target": str(target_path),
        "output": str(output_path),
        "source_bad_case_count": len(source_items),
        "normalized_case_count": len(promoted_items),
        "target_case_count": len(target_items),
        "merged_case_count": len(merged_items),
        "conflict": args.conflict,
        "stats": stats,
        "merged_case_ids": [str(item.get("case_id") or "") for item in merged_items if str(item.get("case_id") or "")],
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        write_json_list(output_path, merged_items)
    return plan


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="Promote reviewed bad cases into formal regression datasets.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE.relative_to(PROJECT_ROOT)), help="Bad case JSON 数组文件")
    parser.add_argument("--target", required=True, help="正式回归集路径，例如 eval_sets/business_depth_regression.json")
    parser.add_argument("--output", default="", help="输出路径；默认覆盖 target")
    parser.add_argument(
        "--conflict",
        choices=("replace", "skip", "error"),
        default="replace",
        help="当 target 已存在同名 case_id 时如何处理",
    )
    parser.add_argument("--dry-run", action="store_true", help="只生成合并计划，不写文件")
    return parser


def main() -> None:
    """命令行入口。"""

    configure_utf8_stdio()
    args = build_parser().parse_args()
    plan = build_plan(args)
    print_json(plan)


if __name__ == "__main__":
    main()
