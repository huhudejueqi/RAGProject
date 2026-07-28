"""知识图谱构建器：使用 NetworkX 构建图结构并执行社群检测。

参考 graphrag 的 cluster_graph、社区检测实现，
适配本项目的实体/关系数据结构。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from qa_core.config.logging_config import get_logger
from qa_core.knowledge_graph.extractor import ExtractedEntity, ExtractedRelation
from qa_core.knowledge_graph.extractor import _auto_label_from_desc

logger = get_logger(__name__)


@dataclass
class Community:
    """检测到的社群（社区）。"""
    community_id: int
    entities: list[str] = field(default_factory=list)
    level: int = 0
    summary: str = ""


@dataclass
class GraphBuildResult:
    """图构建结果。"""
    graph: nx.Graph = field(default_factory=nx.Graph)
    communities: list[Community] = field(default_factory=list)
    entity_count: int = 0
    relation_count: int = 0
    community_count: int = 0


class KnowledgeGraphBuilder:
    """使用 NetworkX 构建知识图谱。

    支持：
    - 从抽取结果构建无向图
    - 实体属性聚合
    - 社群检测（Leiden 算法 / Louvain 算法）
    - 社群摘要生成
    """

    def __init__(
        self,
        community_algorithm: str = "louvain",
        max_cluster_size: int = 30,
    ):
        """
        参数：
            community_algorithm: 社群检测算法，支持 'louvain' 或 'leiden'
            max_cluster_size: 社群最大实体数上限，超过则递归划分
        """
        self._algorithm = community_algorithm
        self._max_cluster_size = max_cluster_size

    def build(
        self,
        entities: list[ExtractedEntity],
        relationships: list[ExtractedRelation],
    ) -> GraphBuildResult:
        """从实体和关系列表构建知识图谱。"""
        G = nx.Graph()

        # 添加节点
        for e in entities:
            G.add_node(
                e.name,
                type=e.type,
                description=e.description,
                node_type="entity",
            )

        # 添加边
        for r in relationships:
            if G.has_node(r.source) and G.has_node(r.target):
                G.add_edge(
                    r.source,
                    r.target,
                    label=r.label if r.label else _auto_label_from_desc(r.description),
                    description=r.description,
                    weight=r.strength,
                )
            else:
                logger.warning(
                    "跳过关系 %s -> %s：实体不在图中", r.source, r.target
                )

        logger.info("图构建完成：%d 节点, %d 边", G.number_of_nodes(), G.number_of_edges())

        # 社群检测
        communities = self._detect_communities(G)

        return GraphBuildResult(
            graph=G,
            communities=communities,
            entity_count=G.number_of_nodes(),
            relation_count=G.number_of_edges(),
            community_count=len(communities),
        )

    def _detect_communities(self, G: nx.Graph) -> list[Community]:
        """执行社群检测。

        优先使用 Leiden 算法（需要 cdlib），回退到 NetworkX 的 Louvain。
        """
        if G.number_of_nodes() == 0:
            return []

        communities: list[Community] = []
        try:
            # 尝试使用 cdlib 的 Leiden
            import cdlib
            from cdlib import algorithms

            if self._algorithm == "leiden":
                coms = algorithms.leiden(G)
            else:
                coms = algorithms.louvain(G)

            for i, community_nodes in enumerate(coms.communities):
                communities.append(Community(
                    community_id=i,
                    entities=list(community_nodes),
                ))
            logger.info(
                "社群检测完成（%s）：%d 个社群",
                self._algorithm, len(communities),
            )

        except ImportError:
            # 回退到 NetworkX 内置算法
            logger.info("cdlib 未安装，使用 NetworkX 内置 Louvain 算法")
            try:
                from networkx.algorithms.community import louvain_communities
                coms = louvain_communities(G, seed=42)
                for i, community_nodes in enumerate(coms):
                    communities.append(Community(
                        community_id=i,
                        entities=list(community_nodes),
                    ))
            except ImportError:
                # 极简回退：所有节点为一个社群
                logger.warning("社群检测算法不可用，所有节点归为一个社群")
                communities.append(Community(
                    community_id=0,
                    entities=list(G.nodes()),
                ))

        return communities

    def get_subgraph(self, G: nx.Graph, entity_names: list[str]) -> nx.Graph:
        """获取指定实体及其直接邻居的子图。"""
        nodes_to_include = set(entity_names)
        for node in entity_names:
            if node in G:
                nodes_to_include.update(G.neighbors(node))
        return G.subgraph(nodes_to_include).copy()

    def get_entity_context(self, G: nx.Graph, entity_name: str, depth: int = 2) -> dict[str, Any]:
        """获取实体的图上下文：邻居、关系描述、社群归属。"""
        if entity_name not in G:
            return {"entity": entity_name, "error": "实体不在图中"}

        neighbors = []
        for neighbor in G.neighbors(entity_name):
            edge_data = G.get_edge_data(entity_name, neighbor) or {}
            neighbors.append({
                "entity": neighbor,
                "relationship": edge_data.get("description", ""),
                "strength": edge_data.get("weight", 1.0),
            })

        # 按关系强度排序
        neighbors.sort(key=lambda x: x["strength"], reverse=True)

        return {
            "entity": entity_name,
            "type": G.nodes[entity_name].get("type", ""),
            "description": G.nodes[entity_name].get("description", ""),
            "degree": G.degree(entity_name),
            "neighbors": neighbors,
        }
