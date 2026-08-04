"""知识图谱查询辅助函数测试。"""

from qa_core.api.graph import _graph_search_keys


def test_graph_search_keys_extract_entity_from_question() -> None:
    """验证自然语言问题会保留完整问题并追加实体词。"""
    keys = _graph_search_keys("糖尿病怎么办")
    assert keys == ["糖尿病怎么办", "糖尿病"]


def test_graph_search_keys_handles_question_marker_at_start() -> None:
    """验证“怎么治疗糖尿病”也能拆出“糖尿病”。"""
    keys = _graph_search_keys("怎么治疗糖尿病")
    assert "糖尿病" in keys


def test_graph_search_keys_keeps_direct_entity() -> None:
    """验证短实体查询不需要额外拆分。"""
    assert _graph_search_keys("腹部不适") == ["腹部不适"]
