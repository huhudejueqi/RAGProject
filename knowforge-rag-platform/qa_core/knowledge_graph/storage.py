"""知识图谱存储层：实体、关系、社群持久化到 Milvus。

利用项目中已有的 Milvus 基础设施，将抽取的实体和关系
存储为独立的 Milvus Collection，支持向量检索和属性过滤。
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from pymilvus import DataType, FieldSchema, MilvusClient

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings

logger = get_logger(__name__)

# ── 集合名称常量 ──
ENTITY_COLLECTION = "kg_entities"
RELATION_COLLECTION = "kg_relations"
COMMUNITY_COLLECTION = "kg_communities"

EMBEDDING_DIM = 1024  # BGE-M3 固定 1024 维

class GraphStorage:
    """知识图谱 Milvus 存储。

    管理三个集合：
    - kg_entities: 实体（含 description 向量）
    - kg_relations: 关系（含描述文本）
    - kg_communities: 社群（含摘要）
    """

    def __init__(self, collection_name_prefix: str = ""):
        self._prefix = collection_name_prefix
        self._client: MilvusClient | None = None
        self._settings = get_settings()

    @property
    def _entity_collection(self) -> str:
        return f"{self._prefix}{ENTITY_COLLECTION}"

    @property
    def _relation_collection(self) -> str:
        return f"{self._prefix}{RELATION_COLLECTION}"

    @property
    def _community_collection(self) -> str:
        return f"{self._prefix}{COMMUNITY_COLLECTION}"

    def _get_client(self) -> MilvusClient:
        """获取或创建 MilvusClient。"""
        if self._client is None:
            uri = self._settings.milvus_uri
            db_name = self._settings.milvus_database
            self._client = MilvusClient(uri=uri, db_name=db_name)
            try:
                from pymilvus import connections
                connections.connect(uri=uri, db_name=db_name or "")
            except Exception:
                pass
        return self._client

    def ensure_collections(self, dim: int = 1024):
        """确保知识图谱的三个集合已创建。"""
        client = self._get_client()

        # 实体集合
        if not client.has_collection(self._entity_collection):
            schema = MilvusClient.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )
            schema.add_field(FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True))
            schema.add_field(FieldSchema(name="entity_id", dtype=DataType.VARCHAR, max_length=256))
            schema.add_field(FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=512))
            schema.add_field(FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=128))
            schema.add_field(FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096))
            schema.add_field(FieldSchema(name="description_vector", dtype=DataType.FLOAT_VECTOR, dim=dim))

            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="description_vector",
                metric_type="IP",
                index_type="HNSW",
                params={"M": 16, "efConstruction": 200},
            )
            index_params.add_index(field_name="name", index_type="Trie")

            client.create_collection(
                collection_name=self._entity_collection,
                schema=schema,
                index_params=index_params,
            )
            logger.info("创建实体集合: %s (dim=%d)", self._entity_collection, dim)

        # 关系集合
        if not client.has_collection(self._relation_collection):
            schema = MilvusClient.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )
            schema.add_field(FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True))
            schema.add_field(FieldSchema(name="relation_id", dtype=DataType.VARCHAR, max_length=256))
            schema.add_field(FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512))
            schema.add_field(FieldSchema(name="target", dtype=DataType.VARCHAR, max_length=512))
            schema.add_field(FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096))
            schema.add_field(FieldSchema(name="strength", dtype=DataType.FLOAT))

            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(field_name="source", index_type="Trie")
            index_params.add_index(field_name="target", index_type="Trie")

            client.create_collection(
                collection_name=self._relation_collection,
                schema=schema,
                index_params=index_params,
            )
            logger.info("创建关系集合: %s", self._relation_collection)

        # 社群集合
        if not client.has_collection(self._community_collection):
            schema = MilvusClient.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )
            schema.add_field(FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True))
            schema.add_field(FieldSchema(name="community_id", dtype=DataType.INT64))
            schema.add_field(FieldSchema(name="entities", dtype=DataType.VARCHAR, max_length=8192))  # JSON list
            schema.add_field(FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=4096))

            client.create_collection(
                collection_name=self._community_collection,
                schema=schema,
            )
            logger.info("创建社群集合: %s", self._community_collection)

    def store_graph(
        self,
        result: GraphBuildResult,
        kb_version: str = "",
    ) -> dict[str, int]:
        """将完整的图构建结果存入 Milvus。"""
        client = self._get_client()

        entity_count = 0
        rel_count = 0
        com_count = 0

        # 写入实体
        if result.entity_count > 0:
            entities_data = []
            for node, data in result.graph.nodes(data=True):
                entities_data.append({
                    "entity_id": str(uuid.uuid4()),
                    "name": node,
                    "type": data.get("type", "OTHER"),
                    "description": data.get("description", ""),
                    "kb_version": kb_version,
                })
            if entities_data:
                client.insert(self._entity_collection, entities_data)
                entity_count = len(entities_data)

        # 写入关系
        if result.relation_count > 0:
            relations_data = []
            for u, v, data in result.graph.edges(data=True):
                relations_data.append({
                    "relation_id": str(uuid.uuid4()),
                    "source": u,
                    "target": v,
                    "description": data.get("description", ""),
                    "strength": data.get("weight", 1.0),
                    "kb_version": kb_version,
                })
            if relations_data:
                client.insert(self._relation_collection, relations_data)
                rel_count = len(relations_data)

        # 写入社群
        if result.communities:
            communities_data = []
            for com in result.communities:
                communities_data.append({
                    "community_id": com.community_id,
                    "entities": json.dumps(com.entities, ensure_ascii=False),
                    "summary": com.summary or "",
                    "kb_version": kb_version,
                })
            if communities_data:
                client.insert(self._community_collection, communities_data)
                com_count = len(communities_data)

        logger.info(
            "图存储完成: %d 实体, %d 关系, %d 社群",
            entity_count, rel_count, com_count,
        )
        return {
            "entities": entity_count,
            "relationships": rel_count,
            "communities": com_count,
        }

    def search_entities(
        self,
        query: str,
        top_k: int = 10,
        entity_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """按实体名称或类型搜索实体。"""
        client = self._get_client()

        expr = f'name like "%{query}%"'
        if entity_types:
            type_list = '", "'.join(entity_types)
            expr = f'{expr} and type in ["{type_list}"]'

        results = client.query(
            collection_name=self._entity_collection,
            filter=expr,
            limit=top_k,
            output_fields=["name", "type", "description"],
        )
        return results

    def get_entity_relations(
        self,
        entity_name: str,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """获取指定实体的所有关系。"""
        client = self._get_client()

        results = client.query(
            collection_name=self._relation_collection,
            filter=f'source == "{entity_name}" or target == "{entity_name}"',
            limit=top_k,
            output_fields=["source", "target", "description", "strength"],
        )
        return results

    def drop_collections(self):
        """删除所有知识图谱集合（用于测试或重建）。"""
        client = self._get_client()
        for col in [self._entity_collection, self._relation_collection, self._community_collection]:
            if client.has_collection(col):
                client.drop_collection(col)
                logger.info("已删除集合: %s", col)
