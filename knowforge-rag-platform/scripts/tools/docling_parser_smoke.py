"""Docling 文档解析增强后端冒烟检查。

该脚本只验证 `DOCUMENT_PARSER_BACKEND=docling` 的加载与解析能力，不写入 Milvus，
也不创建知识库版本。默认会生成一个临时 HTML 样例，方便在新环境中快速确认 Docling
是否安装可用；也可以通过 `--input` 指定真实 PDF、DOCX、PPTX 或 HTML 文件。

用法：
    python scripts/tools/docling_parser_smoke.py
    python scripts/tools/docling_parser_smoke.py --input 你的复杂版面资料.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qa_core.config.settings import get_settings  # noqa: E402
from qa_core.indexing.document_loaders import load_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。

    调用顺序：命令行入口 -> build_parser()。
    """
    parser = argparse.ArgumentParser(description="Smoke test Docling document parser backend.")
    parser.add_argument("--input", default="", help="可选：指定一个 PDF/DOCX/PPTX/HTML 文件。")
    parser.add_argument("--preview-chars", type=int, default=600, help="输出正文预览长度。")
    return parser


def create_sample_html(directory: Path) -> Path:
    """生成一个最小 HTML 样例，用于不依赖业务资料的解析验证。

    调用顺序：命令行入口 -> create_sample_html()。
    """
    path = directory / "docling_smoke_sample.html"
    path.write_text(
        "\n".join(
            [
                "<html><body>",
                "<h1>Docling parser smoke</h1>",
                "<p>预算审批超过 8000 元时，需要补充预算占用说明。</p>",
                "<table>",
                "<tr><th>材料</th><th>状态</th></tr>",
                "<tr><td>预算占用记录</td><td>必填</td></tr>",
                "</table>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return path


def resolve_input(value: str, temp_dir: Path) -> Path:
    """解析输入文件；未传入时生成临时 HTML 样例。

    调用顺序：命令行入口 -> resolve_input()。
    """
    if not value:
        return create_sample_html(temp_dir)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在：{path}")
    return path


def main() -> None:
    """运行 Docling loader 冒烟检查。

    调用顺序：命令行入口 -> main()。
    """
    parser = build_parser()
    args = parser.parse_args()
    os.environ["DOCUMENT_PARSER_BACKEND"] = "docling"

    get_settings.cache_clear()

    with tempfile.TemporaryDirectory() as temp_dir:
        path = resolve_input(args.input, Path(temp_dir))
        try:
            documents = load_file(path)
        except RuntimeError as exc:
            print(
                {
                    "ok": False,
                    "parser_backend": get_settings().document_parser_backend,
                    "input": str(path),
                    "error": str(exc),
                }
            )
            raise SystemExit(1) from exc

    content = "\n\n".join(document.page_content for document in documents).strip()
    metadata = [document.metadata for document in documents]
    print(
        {
            "ok": bool(content),
            "parser_backend": get_settings().document_parser_backend,
            "input": str(path),
            "document_count": len(documents),
            "metadata": metadata,
            "preview": content[: args.preview_chars],
        }
    )


if __name__ == "__main__":
    main()
