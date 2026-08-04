"""社区摘要生成与社区级搜索测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import networkx as nx

from qa_core.knowledge_graph.community_search import (
    CommunitySummarizer,
    build_community_context,
    global_search,
    score_communities,
)
from qa_core.knowledge_graph.graph_builder import Community


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    async def ainvoke(self, messages):
        return SimpleNamespace(content=self._content)


class CommunityContextTests(unittest.TestCase):
    def test_build_community_context_contains_entities_and_relations(self) -> None:
        graph = nx.Graph()
        graph.add_node("人事部", type="组织", description="负责入职流程")
        graph.add_node("新员工", type="人物", description="办理入职手续")
        graph.add_edge("人事部", "新员工", label="办理", description="人事部为新员工办理入职")

        entity_text, relation_text = build_community_context(
            graph, Community(community_id=0, entities=["人事部", "新员工"])
        )

        self.assertIn("人事部", entity_text)
        self.assertIn("新员工", entity_text)
        self.assertIn("办理", relation_text)

    def test_build_community_context_respects_char_budget(self) -> None:
        graph = nx.Graph()
        graph.add_node("人事部", type="组织", description="很长的描述" * 100)
        graph.add_node("新员工", type="人物", description="另一段很长的描述" * 100)
        graph.add_edge("人事部", "新员工", label="办理", description="很长的关系描述" * 100)

        entity_text, relation_text = build_community_context(
            graph,
            Community(community_id=0, entities=["人事部", "新员工"]),
            max_context_chars=50,
        )

        self.assertLessEqual(len(entity_text) + len(relation_text), 50)

    def test_score_communities_ranks_relevant_summary_first(self) -> None:
        communities = [
            {"community_id": 1, "summary": "员工入职流程", "entities": "人事部"},
            {"community_id": 2, "summary": "员工入职合同签署流程", "entities": "人事部"},
        ]

        scored = score_communities("员工入职合同", communities)

        self.assertGreater(scored[0][0], scored[1][0])
        self.assertEqual(scored[0][1]["community_id"], 2)


class CommunitySummarizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_summarize_returns_llm_summary(self) -> None:
        graph = nx.Graph()
        graph.add_node("人事部", type="组织", description="负责入职流程")
        graph.add_node("新员工", type="人物", description="办理入职手续")
        graph.add_edge("人事部", "新员工", label="办理", description="人事部为新员工办理入职")
        community = Community(community_id=0, entities=["人事部", "新员工"])
        summarizer = CommunitySummarizer(llm=_FakeLLM("人事社区负责员工入职流程"))

        summary = await summarizer.summarize(graph, community)

        self.assertEqual(summary, "人事社区负责员工入职流程")


class GlobalSearchTests(unittest.TestCase):
    def test_global_search_formats_relevant_community(self) -> None:
        class FakeStorage:
            def search_communities(self, max_results=200):
                return [
                    {
                        "community_id": 2,
                        "entities": "人事部,新员工",
                        "summary": "员工入职和合同签署流程",
                    }
                ]

        text = global_search("员工入职", storage=FakeStorage())

        self.assertIn("[社区知识参考]", text)
        self.assertIn("员工入职和合同签署流程", text)
