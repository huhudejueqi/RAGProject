"""验证 BERT 意图模型加载、预测和离线准确率。

调用顺序：课程示例、测试或命令行入口 -> 本模块公开接口。
"""

from __future__ import annotations

import unittest

from qa_core.intent.model_classifier import DEFAULT_EVAL_EXAMPLES, RETRIEVAL_INTENTS, BertIntentModelService


class TestBertIntentModelService(unittest.TestCase):
    """验证本地 BERT 意图模型服务的运行契约。

    调用顺序：业务模块 -> TestBertIntentModelService。
    """

    @classmethod
    def setUpClass(cls) -> None:
        """加载当前测试类共享的模型制品。

        调用顺序：测试或业务入口 -> TestBertIntentModelService.setUpClass()。
        """
        cls.service = BertIntentModelService.from_settings()

    def test_loads_expected_label_mapping(self) -> None:
        """验证模型标签顺序与训练输出保持一致。

        调用顺序：测试或业务入口 -> TestBertIntentModelService.test_loads_expected_label_mapping()。
        """
        self.assertEqual(self.service.labels, RETRIEVAL_INTENTS)
        self.assertEqual(self.service.model_version, "bert-intent-v1")

    def test_evaluation_has_usable_accuracy_for_demo_dataset(self) -> None:
        """验证模型在冻结样例集上达到最低准确率。

        调用顺序：测试或业务入口 -> TestBertIntentModelService.test_evaluation_has_usable_accuracy_for_demo_dataset()。
        """
        evaluation = self.service.evaluate(DEFAULT_EVAL_EXAMPLES)

        self.assertGreaterEqual(evaluation.accuracy, 0.75)
        self.assertEqual(evaluation.total, len(DEFAULT_EVAL_EXAMPLES))

    def test_predicts_faq_query(self) -> None:
        """验证标准流程问题识别为 FAQ_QUERY。

        调用顺序：测试或业务入口 -> TestBertIntentModelService.test_predicts_faq_query()。
        """
        prediction = self.service.predict("新人入职流程有哪些")

        self.assertEqual(prediction.intent, "FAQ_QUERY")
        self.assertGreater(prediction.score, 0.5)

    def test_predicts_knowledge_query(self) -> None:
        """验证制度解释问题识别为 KNOWLEDGE_QUERY。

        调用顺序：测试或业务入口 -> TestBertIntentModelService.test_predicts_knowledge_query()。
        """
        prediction = self.service.predict("请解释员工离职权限回收制度")

        self.assertEqual(prediction.intent, "KNOWLEDGE_QUERY")
        self.assertGreater(prediction.score, 0.5)

    def test_history_feature_supports_follow_up(self) -> None:
        """验证历史特征能够支持 FOLLOW_UP 识别。

        调用顺序：测试或业务入口 -> TestBertIntentModelService.test_history_feature_supports_follow_up()。
        """
        prediction = self.service.predict("那要谁审批", has_history=True)

        self.assertEqual(prediction.intent, "FOLLOW_UP")
        self.assertGreater(prediction.score, 0.5)


if __name__ == "__main__":
    unittest.main()
