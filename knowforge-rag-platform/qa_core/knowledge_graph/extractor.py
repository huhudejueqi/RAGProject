"""LLM 驱动的实体与关系抽取器。

参考 graphrag-Chinese-llm 的 GraphExtractor 实现，
适配本项目的 LangChain ChatOpenAI 调用方式。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.knowledge_graph.prompts import (
    CONTINUE_EXTRACT_PROMPT,
    DEFAULT_COMPLETION_DELIMITER,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_RECORD_DELIMITER,
    DEFAULT_TUPLE_DELIMITER,
    ENTITY_RELATION_EXTRACT_PROMPT,
    SUMMARIZE_DESCRIPTIONS_PROMPT,
)

logger = get_logger(__name__)


@dataclass
class ExtractedEntity:
    """从文本中抽取的实体。"""
    name: str
    type: str
    description: str
    source_doc_id: str = ""
    source_chunk_id: str = ""


@dataclass
class ExtractedRelation:
    """从文本中抽取的关系。"""
    source: str
    target: str
    description: str
    strength: float = 1.0
    source_doc_id: str = ""
    source_chunk_id: str = ""


@dataclass
class ExtractionResult:
    """单次文本抽取的结果。"""
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelation] = field(default_factory=list)


class GraphExtractor:
    """LLM 驱动的实体与关系抽取器。

    使用 LLM 从文档文本块中提取实体和关系，
    支持多轮迭代抽取（gleaning）以提高召回率。

    用法：
        extractor = GraphExtractor()
        result = await extractor.extract(chunk_text)
    """

    def __init__(
        self,
        entity_types: list[str] | None = None,
        tuple_delimiter: str = DEFAULT_TUPLE_DELIMITER,
        record_delimiter: str = DEFAULT_RECORD_DELIMITER,
        completion_delimiter: str = DEFAULT_COMPLETION_DELIMITER,
        max_gleanings: int = 1,
    ):
        self._entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self._tuple_delimiter = tuple_delimiter
        self._record_delimiter = record_delimiter
        self._completion_delimiter = completion_delimiter
        self._max_gleanings = max_gleanings
        self._llm = None

    def _get_llm(self):
        """懒加载 LLM 客户端。"""
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            settings = get_settings()
            self._llm = ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                temperature=0.1,  # 抽取任务使用低温度
                timeout=120,
                streaming=False,
            )
        return self._llm

    def _build_extract_prompt(self, text: str) -> str:
        """构建实体关系抽取提示。"""
        return ENTITY_RELATION_EXTRACT_PROMPT.format(
            entity_types=", ".join(self._entity_types),
            input_text=text,
            tuple_delimiter=self._tuple_delimiter,
            record_delimiter=self._record_delimiter,
            completion_delimiter=self._completion_delimiter,
        )

    def _build_continue_prompt(self, text: str, existing_output: str) -> str:
        """构建继续抽取提示。"""
        return CONTINUE_EXTRACT_PROMPT.format(
            entity_types=", ".join(self._entity_types),
            input_text=text,
            existing_output=existing_output,
            completion_delimiter=self._completion_delimiter,
        )

    def _parse_response(self, response: str) -> ExtractionResult:
        """解析 LLM 返回的结构化文本为实体和关系列表。"""
        entities: list[ExtractedEntity] = []
        relationships: list[ExtractedRelation] = []

        # 按 record_delimiter 拆分
        records = response.split(self._record_delimiter)
        for record in records:
            record = record.strip()
            if not record:
                continue
            if record.startswith("(") and record.endswith(")"):
                record = record[1:-1]
            parts = [p.strip() for p in record.split(self._tuple_delimiter)]
            if not parts:
                continue

            record_type = parts[0].strip('"\'')
            if record_type == "entity" and len(parts) >= 4:
                name = parts[1].strip('"\'')
                etype = parts[2].strip('"\'')
                desc = parts[3].strip('"\'')
                if name:
                    entities.append(ExtractedEntity(
                        name=name.upper(),
                        type=etype.upper(),
                        description=desc,
                    ))
            elif record_type == "relationship" and len(parts) >= 5:
                source = parts[1].strip('"\'')
                target = parts[2].strip('"\'')
                desc = parts[3].strip('"\'')
                try:
                    strength = float(parts[4].strip('"\''))
                except (ValueError, IndexError):
                    strength = 1.0
                if source and target:
                    relationships.append(ExtractedRelation(
                        source=source.upper(),
                        target=target.upper(),
                        description=desc,
                        strength=min(strength, 10.0),
                    ))

        return ExtractionResult(entities=entities, relationships=relationships)

    async def extract(self, text: str) -> ExtractionResult:
        """从文本中提取实体和关系。

        执行一次主抽取 + 最多 max_gleanings 次迭代补充抽取。
        """
        if not text or not text.strip():
            return ExtractionResult()

        llm = self._get_llm()

        # 第一轮：主抽取
        prompt = self._build_extract_prompt(text)
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = str(getattr(response, "content", "") or "")

        if not content.strip():
            logger.warning("LLM 返回空内容，跳过抽取")
            return ExtractionResult()

        # 解析第一轮结果
        result = self._parse_response(content)
        logger.info(
            "第一轮抽取：%d 实体, %d 关系",
            len(result.entities), len(result.relationships),
        )

        # 后续轮次：gleaning（补充抽取）
        full_output = content
        for glean_idx in range(self._max_gleanings):
            if self._completion_delimiter in content:
                logger.debug("第 %d 轮 gleaning 已完成", glean_idx + 1)
                break

            continue_prompt = self._build_continue_prompt(text, full_output)
            response = await llm.ainvoke([HumanMessage(content=continue_prompt)])
            content = str(getattr(response, "content", "") or "")

            if not content.strip():
                break

            # 如果返回的是完成标记，停止
            if content.strip() == self._completion_delimiter:
                break

            # 解析补充结果并合并
            extra = self._parse_response(content)
            result.entities.extend(extra.entities)
            result.relationships.extend(extra.relationships)
            full_output += f"\n{content}"
            logger.info(
                "Gleaning 第 %d 轮：新增 %d 实体, %d 关系",
                glean_idx + 1, len(extra.entities), len(extra.relationships),
            )

        # 去重
        result = self._deduplicate(result)
        logger.info(
            "抽取完成（去重后）：%d 实体, %d 关系",
            len(result.entities), len(result.relationships),
        )
        return result

    def _deduplicate(self, result: ExtractionResult) -> ExtractionResult:
        """按实体名称去重，合并描述。"""
        seen_entities: dict[str, ExtractedEntity] = {}
        for e in result.entities:
            key = e.name.upper()
            if key in seen_entities:
                existing = seen_entities[key]
                if e.description and e.description not in existing.description:
                    existing.description += f"；{e.description}"
            else:
                seen_entities[key] = e

        seen_rels: set[tuple[str, str]] = set()
        unique_rels: list[ExtractedRelation] = []
        for r in result.relationships:
            key = (r.source.upper(), r.target.upper())
            rev_key = (r.target.upper(), r.source.upper())
            if key not in seen_rels and rev_key not in seen_rels:
                seen_rels.add(key)
                unique_rels.append(r)

        return ExtractionResult(
            entities=list(seen_entities.values()),
            relationships=unique_rels,
        )
