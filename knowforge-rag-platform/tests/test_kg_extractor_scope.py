"""知识图谱实体类型白名单与悬空关系测试。"""

from __future__ import annotations

import unittest

from qa_core.knowledge_graph.extractor import ExtractedEntity, ExtractedRelation, GraphExtractor
from qa_core.knowledge_graph.graph_builder import KnowledgeGraphBuilder


class GraphExtractorScopeTests(unittest.TestCase):
    """验证 LLM 解析阶段会丢弃预设实体类型之外的实体。"""

    def test_parse_response_filters_entity_types_outside_preset(self) -> None:
        extractor = GraphExtractor(entity_types=["人物"])
        response = (
            '("entity"<|>萧炎<|>人物<|>斗气大陆的修炼者)'
            '##'
            '("entity"<|>斗气<|>功法<|>不在预设类型中)'
        )

        result = extractor._parse_response(response)

        self.assertEqual(len(result.entities), 1)
        self.assertEqual(result.entities[0].name, "萧炎")
        self.assertEqual(result.entities[0].type, "人物")
        self.assertEqual(result.relationships, [])

    def test_entity_type_matching_is_case_insensitive(self) -> None:
        extractor = GraphExtractor(entity_types=["organization"])
        response = '("entity"<|>Acme Corp<|>ORGANIZATION<|>company)'

        result = extractor._parse_response(response)

        self.assertEqual(result.entities[0].name, "ACME CORP")
        self.assertEqual(result.entities[0].type, "ORGANIZATION")


class KnowledgeGraphBuilderOrphanTests(unittest.TestCase):
    """验证关系两端没有对应实体时不会建边。"""

    def test_builder_drops_relationship_with_missing_endpoint(self) -> None:
        entities = [
            ExtractedEntity(name="萧炎", type="人物", description="斗气大陆的修炼者"),
        ]
        relationships = [
            ExtractedRelation(
                source="萧炎",
                target="药老",
                label="师徒",
                description="萧炎是药老的弟子",
            ),
        ]

        result = KnowledgeGraphBuilder().build(entities, relationships)

        self.assertEqual(result.entity_count, 1)
        self.assertEqual(result.relation_count, 0)
