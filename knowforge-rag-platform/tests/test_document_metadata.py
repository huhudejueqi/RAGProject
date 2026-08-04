"""文档元数据展示函数的单元测试。"""

from qa_core.document_metadata import format_source_label


def test_format_source_label_prefers_markdown_section_over_file_name() -> None:
    """验证 Markdown 多章节文档优先展示具体章节标题。"""
    metadata = {
        "file_name": "药品数据.md",
        "h1": "常用药品与适应症参考",
        "h2": "络瘀通胶囊",
    }
    assert format_source_label(metadata) == "络瘀通胶囊"


def test_format_source_label_prefers_deepest_markdown_heading() -> None:
    """验证 h3 比 h2/h1 更具体时优先展示 h3。"""
    metadata = {"file_name": "指南.md", "h1": "总览", "h2": "流程", "h3": "故障处理"}
    assert format_source_label(metadata) == "故障处理"


def test_format_source_label_falls_back_to_file_name_without_heading() -> None:
    """验证没有 Markdown 标题时仍按原逻辑展示文件名。"""
    assert format_source_label({"file_name": "入职流程.pdf"}) == "入职流程.pdf"
