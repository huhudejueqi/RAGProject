"""最终答案置信度测试。"""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from qa_core.pipeline.confidence import (
    calculate_answer_confidence,
    faq_exact_match,
    normalize_retrieval_score,
)


class AnswerConfidenceTests(unittest.TestCase):
    """验证 answer_confidence 和检索 score 分离。

    调用顺序：pytest/unittest 测试入口 -> AnswerConfidenceTests。
    """

    def test_faq_exact_match_has_high_confidence(self) -> None:
        """验证 FAQ 标准问题精确匹配会得到高置信度。

        调用顺序：pytest/unittest 测试入口 -> AnswerConfidenceTests.test_faq_exact_match_has_high_confidence()。
        """
        result = calculate_answer_confidence(
            hit_type="faq_direct",
            retrieval_top_score=0.35,
            context_count=1,
            source_count=1,
            intent_rule_score=0.98,
            query="员工报销需要准备哪些材料？",
            raw_query="员工报销需要准备哪些材料？",
            rewritten_query="员工报销需要准备哪些材料？",
            faq_exact_match=True,
        )

        self.assertEqual(result.level, "high")
        self.assertEqual(result.score, 0.95)
        self.assertIn("faq_exact_match", result.reasons)

    def test_history_rewrite_lowers_but_does_not_replace_retrieval_signal(self) -> None:
        """验证追问改写会降低最终置信度，但不改变检索排序分。

        调用顺序：pytest/unittest 测试入口 -> AnswerConfidenceTests.test_history_rewrite_lowers_but_does_not_replace_retrieval_signal()。
        """
        normal = calculate_answer_confidence(
            hit_type="rag",
            retrieval_top_score=0.8,
            context_count=3,
            source_count=2,
            intent_rule_score=0.84,
            query="新人入职流程怎么走？",
            raw_query="新人入职流程怎么走？",
            rewritten_query="新人入职流程怎么走？",
        )
        rewritten = calculate_answer_confidence(
            hit_type="rag",
            retrieval_top_score=0.8,
            context_count=3,
            source_count=2,
            intent_rule_score=0.84,
            query="这个呢？",
            raw_query="这个呢？",
            rewritten_query="新人入职流程怎么走？",
        )

        self.assertLess(rewritten.score, normal.score)
        self.assertEqual(rewritten.signals["normalized_retrieval_score"], normal.signals["normalized_retrieval_score"])
        self.assertIn("history_rewrite_used", rewritten.reasons)

    def test_insufficient_context_confidence_is_low(self) -> None:
        """验证无上下文分支会输出低置信度。

        调用顺序：pytest/unittest 测试入口 -> AnswerConfidenceTests.test_insufficient_context_confidence_is_low()。
        """
        result = calculate_answer_confidence(
            hit_type="insufficient_context",
            retrieval_top_score=0.2,
            context_count=0,
            source_count=0,
            intent_rule_score=0.6,
            query="完全没有资料的问题",
            raw_query="完全没有资料的问题",
            rewritten_query="完全没有资料的问题",
        )

        self.assertEqual(result.level, "low")
        self.assertIn("insufficient_context", result.reasons)

    def test_faq_exact_match_compares_standard_question(self) -> None:
        """验证 FAQ 精确匹配只看标准问题字段。

        调用顺序：pytest/unittest 测试入口 -> AnswerConfidenceTests.test_faq_exact_match_compares_standard_question()。
        """
        doc = Document(
            page_content="FAQ content",
            metadata={"standard_question": "是否支持开发票？", "answer": "支持。"},
        )

        self.assertTrue(faq_exact_match("是否支持开发票？", doc))
        self.assertFalse(faq_exact_match("可以开发票吗？", doc))

    def test_positive_logits_are_smoothed_for_confidence_only(self) -> None:
        """验证大于 1 的重排 logits 只在置信度中平滑压缩。

        调用顺序：pytest/unittest 测试入口 -> AnswerConfidenceTests.test_positive_logits_are_smoothed_for_confidence_only()。
        """
        self.assertAlmostEqual(normalize_retrieval_score(3.0), 0.75)


if __name__ == "__main__":
    unittest.main()
