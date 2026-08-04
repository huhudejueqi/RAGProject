"""通用关系标签兜底测试。"""

from __future__ import annotations

import unittest

from qa_core.knowledge_graph.extractor import _auto_label_from_desc


class AutoLabelFromDescTests(unittest.TestCase):
    """验证缺失关系标签时的通用兜底逻辑。"""

    def test_extracts_role_from_relation_pattern(self) -> None:
        self.assertEqual(_auto_label_from_desc("天蚕土豆是《斗破苍穹》的作者"), "作者")
        self.assertEqual(_auto_label_from_desc("张三是小明的父亲，负责照顾小明"), "父亲")

    def test_matches_common_chinese_relationship_words(self) -> None:
        self.assertEqual(_auto_label_from_desc("张三与李四存在合作关系"), "合作")
        self.assertEqual(_auto_label_from_desc("甲公司采购了乙公司的设备"), "交易")

    def test_extracts_event_name_from_action_phrase(self) -> None:
        self.assertEqual(_auto_label_from_desc("员工参加了入职培训，并完成签到"), "入职培训")

    def test_unknown_description_uses_generic_fallback(self) -> None:
        self.assertEqual(_auto_label_from_desc("A与B之间关系复杂"), "关联")
        self.assertEqual(_auto_label_from_desc(""), "关联")
