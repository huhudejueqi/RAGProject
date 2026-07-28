"""知识图谱存储层：实体、关系、社群持久化到 Milvus。

利用项目中已有的 Milvus 基础设施，将抽取的实体和关系
存储为独立的 Milvus Collection，支持向量检索和属性过滤。
"""

from __future__ import annotations

import json
import uuid
from typing import Any


def _truncate_utf8(text: str, max_bytes: int = 4000) -> str:
    """按 UTF-8 字节数截断字符串。"""
    if not text:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")

from qa_core.knowledge_graph.graph_builder import GraphBuildResult

from pymilvus import DataType, MilvusClient

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
        return self._client

    def ensure_collections(self, dim: int = 1024):
        """确保知识图谱的三个集合已创建。

        Milvus 要求每个集合必须至少有一个向量字段（dim >= 2）。
        关系集合和社群集合没有语义向量，添加 dummy 2维向量字段满足此约束。
        """
        client = self._get_client()

        # 实体集合：含 description_vector 语义向量
        if not client.has_collection(self._entity_collection):
            schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field(field_name="entity_id", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=128)
            schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=4096)
            schema.add_field(field_name="description_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

            idx = MilvusClient.prepare_index_params()
            idx.add_index(field_name="description_vector", metric_type="IP", index_type="HNSW",
                          params={"M": 16, "efConstruction": 200})
            idx.add_index(field_name="name", index_type="Trie")

            client.create_collection(
                collection_name=self._entity_collection,
                schema=schema,
                index_params=idx,
            )
            client.load_collection(self._entity_collection)
            logger.info("创建实体集合: %s (dim=%d)", self._entity_collection, dim)

        # 关系集合：无语义向量，加 dummy_vec 满足 Milvus 约束
        if not client.has_collection(self._relation_collection):
            schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field(field_name="relation_id", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="target", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="label", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=4096)
            schema.add_field(field_name="strength", datatype=DataType.FLOAT)
            schema.add_field(field_name="dummy_vec", datatype=DataType.FLOAT_VECTOR, dim=2)

            idx = MilvusClient.prepare_index_params()
            idx.add_index(field_name="source", index_type="Trie")
            idx.add_index(field_name="target", index_type="Trie")
            idx.add_index(field_name="dummy_vec", index_type="FLAT", metric_type="L2")

            client.create_collection(
                collection_name=self._relation_collection,
                schema=schema,
                index_params=idx,
            )
            client.load_collection(self._relation_collection)
            logger.info("创建关系集合: %s", self._relation_collection)

        # 社群集合：纯标量 + dummy_vec
        if not client.has_collection(self._community_collection):
            schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field(field_name="community_id", datatype=DataType.INT64)
            schema.add_field(field_name="entities", datatype=DataType.VARCHAR, max_length=8192)
            schema.add_field(field_name="summary", datatype=DataType.VARCHAR, max_length=4096)
            schema.add_field(field_name="dummy_vec", datatype=DataType.FLOAT_VECTOR, dim=2)

            idx = MilvusClient.prepare_index_params()
            idx.add_index(field_name="dummy_vec", index_type="FLAT", metric_type="L2")

            client.create_collection(
                collection_name=self._community_collection,
                schema=schema,
                index_params=idx,
            )
            client.load_collection(self._community_collection)
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
                    "description": _truncate_utf8(data.get("description", "") or ""),
                    "kb_version": kb_version,
                    "description_vector": [0.0] * EMBEDDING_DIM,
                })
            if entities_data:
                client.insert(self._entity_collection, entities_data)
                entity_count = len(entities_data)

        # 写入关系（含 dummy_vec 占位）
        if result.relation_count > 0:
            relations_data = []
            for u, v, data in result.graph.edges(data=True):
                label = data.get("label", "")
                if not label or len(label) > 10:
                    label = _auto_edge_label(u, v, data.get("description", ""))
                relations_data.append({
                    "relation_id": str(uuid.uuid4()),
                    "source": u,
                    "target": v,
                    "label": label[:10],
                    "description": _truncate_utf8(data.get("description", "") or ""),
                    "strength": data.get("weight", 1.0),
                    "dummy_vec": [0.0, 0.0],
                    "kb_version": kb_version,
                })
            if relations_data:
                client.insert(self._relation_collection, relations_data)
                rel_count = len(relations_data)

        # 写入社群（含 dummy_vec 占位）
        if result.communities:
            communities_data = []
            for com in result.communities:
                communities_data.append({
                    "community_id": com.community_id,
                    "entities": json.dumps(com.entities, ensure_ascii=False),
                    "summary": com.summary or "",
                    "dummy_vec": [0.0, 0.0],
                    "kb_version": kb_version,
                    "description_vector": [0.0] * EMBEDDING_DIM,
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

def _auto_edge_label(source: str, target: str, desc: str) -> str:
    """从节点名和描述中推断边标签（2-6字）。"""
    return _auto_label_from_desc(desc)

def _auto_label_from_desc(desc: str) -> str:
    """从关系描述中自动推断短标签。"""
    import re
    desc = desc.strip()
    if not desc:
        return "关联"
    m = re.search(r'是.+的(.{2,8})(?:[，。！？、；：]|$)', desc)
    if m:
        rel = re.sub(r'[，。！？、；：""''（）【】《》\s]', '', m.group(1))
        ignore = {'帮助','进行','获得','需要','一个','一种','一些','这个'}
        if 2 <= len(rel) <= 6 and rel not in ignore:
            return rel
    after = re.sub(r'^(?:萧炎|药老|萧战|纳兰|萧薰|萧媚|萧宁|雅妃|加列|云岚|萧家|熏儿|墨管家|三位长老|神秘老者|古戒|戒指|婚约|黑色)', '', desc[:20])
    if after.startswith('与'):
        m = re.search(r'与(.{2,12}?)(?:关系|之间|，|的)', after)
        if m:
            rel = m.group(1)
            if 2 <= len(rel) <= 4:
                return rel
        return "关系"
    if after.startswith('是'):
        after2 = after[1:]
        after2 = re.sub(r'^(?:萧家|云岚|米特尔|加列|魔兽|斗气|炼药|纳兰|迦南)', '', after2)
        m = re.search(r'^(.{2,8})(?:[，。！？、；：]|$)', after2)
        if m:
            rel = m.group(1)
            if 2 <= len(rel) <= 4:
                return rel
    if after.startswith('在'):
        return "所在地"
    if after.startswith(('向', '从', '对', '给')):
        m = re.search(r'[向从对给](.{2,6})', after)
        if m:
            rel = m.group(1).replace(',', '').replace('，', '')
            if 2 <= len(rel) <= 4:
                return rel
        return "关系"
    for kw in ['修炼', '学习', '炼药', '炼制', '研究']:
        if kw in after[:10]:
            return kw
    for kw in ['计划', '前往', '约定', '前往']:
        if kw in after[:10]:
            return kw
    known = {
        '敌对':['敌对','冲突','对抗','打压','竞争'],
        '合作':['合作','联手','合伙'],
        '交易':['拍卖','购买','售卖','交易'],
        '家族':['家族','族员','族人','成员'],
        '好友':['好友','朋友','伙伴','亲密'],
        '父子':['父亲','儿子','父子','父女'],
        '师徒':['师傅','老师','教导','传授','指导','徒弟','弟子'],
    }
    for label, keywords in known.items():
        for kw in keywords:
            if kw in desc[:30]:
                return label
    m = re.search(r'(?:参加|举行|举办|召开)(.{2,8})(?:[，。！？]|$)', desc)
    if m:
        rel = re.sub(r'[，。！？、；：""''（）【】《》\s]', '', m.group(1))
        if 2 <= len(rel) <= 5:
            return rel
    m = re.search(r'测试出(.{2,8})(?:[，。！？]|$)', desc)
    if m:
        rel = re.sub(r'[，。！？、；：""''（）【】《》\s]', '', m.group(1))
        if 2 <= len(rel) <= 5:
            return rel
    return desc[:3]
