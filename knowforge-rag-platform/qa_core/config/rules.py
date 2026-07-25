"""Shared rule configuration loaded from config/rules.toml."""

from __future__ import annotations

import re
import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path

from qa_core.config.settings import PROJECT_ROOT


DEFAULT_RULE_CONFIG_PATH = PROJECT_ROOT / "config" / "rules.toml"


@dataclass(frozen=True)
class FaqFastPathRules:
    """Rules that decide whether a short query is worth probing in FAQ first.

    调用顺序：启动配置或前置校验 -> FaqFastPathRules。
    """

    max_chars: int
    hints: tuple[str, ...]

    def hint_matches(self, query: str) -> bool:
        """Return whether query contains any configured FAQ fast-path hint.

        调用顺序：启动配置或前置校验 -> FaqFastPathRules.hint_matches()。
        """

        if not self.hints:
            return False
        pattern = re.compile("|".join(re.escape(item) for item in self.hints), re.IGNORECASE)
        return bool(pattern.search(query or ""))


@dataclass(frozen=True)
class QueryVariantReplacementRule:
    """One configured deterministic query-variant replacement rule.

    调用顺序：启动配置或前置校验 -> QueryVariantReplacementRule。
    """

    when_any: tuple[str, ...]
    when_all: tuple[str, ...]
    replacements: tuple[tuple[str, str], ...]
    ignore_case: bool = False

    def matches(self, query: str) -> bool:
        """Return whether this replacement rule should run for query.

        调用顺序：启动配置或前置校验 -> QueryVariantReplacementRule.matches()。
        """

        source = query.lower() if self.ignore_case else query
        any_terms = tuple(item.lower() for item in self.when_any) if self.ignore_case else self.when_any
        all_terms = tuple(item.lower() for item in self.when_all) if self.ignore_case else self.when_all
        if any_terms and not any(term in source for term in any_terms):
            return False
        if all_terms and not all(term in source for term in all_terms):
            return False
        return bool(any_terms or all_terms)


@dataclass(frozen=True)
class QueryVariantRules:
    """Configured deterministic query-variant rules.

    调用顺序：启动配置或前置校验 -> QueryVariantRules。
    """

    short_structured_max_chars: int
    short_structured_markers: tuple[str, ...]
    replacements: tuple[QueryVariantReplacementRule, ...]

    def is_short_structured_question(self, query: str) -> bool:
        """Return whether query is short and specific enough to skip expansion.

        调用顺序：启动配置或前置校验 -> QueryVariantRules.is_short_structured_question()。
        """

        compact = query.strip()
        if not compact or len(compact) > self.short_structured_max_chars:
            return False
        return any(marker in compact for marker in self.short_structured_markers)


@dataclass(frozen=True)
class RetrievalStrategyRules:
    """Configured retrieval guardrails shared by V1 and V2.

    调用顺序：启动配置或前置校验 -> RetrievalStrategyRules。
    """

    faq_direct_floor: float
    faq_direct_discount: float
    short_query_guard_threshold: float
    low_rule_score_threshold: float
    medium_rule_score_threshold: float
    medium_rule_score_direct_threshold: float
    pricing_direct_threshold: float
    compliance_direct_threshold: float
    low_rule_score_direct_threshold: float
    follow_up_faq_top_k_min: int
    strong_faq_doc_top_k_min: int
    knowledge_context_top_n_min: int
    guard_context_top_n_min: int
    table_context_top_n_min: int


@dataclass(frozen=True)
class IntentRuleScoreRules:
    """Scores used to rank deterministic intent-rule candidates.

    调用顺序：启动配置或前置校验 -> IntentRuleScoreRules。
    """

    strong_faq: float
    knowledge: float
    source_question_shape: float
    direct_faq_shape: float


@dataclass(frozen=True)
class RuleConfig:
    """Runtime rules shared by pipeline modules.

    调用顺序：启动配置或前置校验 -> RuleConfig。
    """

    faq_fast_path: FaqFastPathRules
    query_variants: QueryVariantRules
    intent_rule_scores: IntentRuleScoreRules
    retrieval_strategy: RetrievalStrategyRules


def get_rule_config(path: str | Path | None = None) -> RuleConfig:
    """Load routing rules from TOML.

    The file is intentionally read on demand so local rule edits take effect on
    the next request or test run, matching the scenario.toml workflow.

    调用顺序：启动配置或前置校验 -> get_rule_config()。
    """

    config_path = Path(path) if path else DEFAULT_RULE_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    faq_payload = dict(payload.get("faq_fast_path") or {})
    max_chars = int(faq_payload.get("max_chars") or 0)
    hints = tuple(str(item).strip() for item in faq_payload.get("hints", ()) if str(item).strip())
    if max_chars <= 0:
        raise ValueError(f"faq_fast_path.max_chars 必须大于 0：{config_path}")
    if not hints:
        raise ValueError(f"faq_fast_path.hints 不能为空：{config_path}")
    query_variants = _load_query_variant_rules(payload, config_path)
    intent_rule_scores = _load_intent_rule_score_rules(payload, config_path)
    retrieval_strategy = _load_retrieval_strategy_rules(payload, config_path)
    return RuleConfig(
        faq_fast_path=FaqFastPathRules(max_chars=max_chars, hints=hints),
        query_variants=query_variants,
        intent_rule_scores=intent_rule_scores,
        retrieval_strategy=retrieval_strategy,
    )


def _load_query_variant_rules(payload: dict, config_path: Path) -> QueryVariantRules:
    """Parse query variant rules from TOML payload.

    调用顺序：启动配置或前置校验 -> _load_query_variant_rules()。
    """

    variant_payload = dict(payload.get("query_variants") or {})
    max_chars = int(variant_payload.get("short_structured_max_chars") or 0)
    markers = _clean_tuple(variant_payload.get("short_structured_markers", ()))
    if max_chars <= 0:
        raise ValueError(f"query_variants.short_structured_max_chars 必须大于 0：{config_path}")
    if not markers:
        raise ValueError(f"query_variants.short_structured_markers 不能为空：{config_path}")

    replacements = tuple(
        _parse_replacement_rule(item, config_path)
        for item in variant_payload.get("replacements", ())
    )
    if not replacements:
        raise ValueError(f"query_variants.replacements 不能为空：{config_path}")

    return QueryVariantRules(
        short_structured_max_chars=max_chars,
        short_structured_markers=markers,
        replacements=replacements,
    )


def _parse_replacement_rule(payload: dict, config_path: Path) -> QueryVariantReplacementRule:
    """Parse one query variant replacement rule.

    调用顺序：启动配置或前置校验 -> _parse_replacement_rule()。
    """

    rule_payload = dict(payload or {})
    when_any = _clean_tuple(rule_payload.get("when_any", ()))
    when_all = _clean_tuple(rule_payload.get("when_all", ()))
    replacements = tuple(
        (str(pair[0]).strip(), str(pair[1]).strip())
        for pair in rule_payload.get("replace", ())
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and str(pair[0]).strip() and str(pair[1]).strip()
    )
    if not when_any and not when_all:
        raise ValueError(f"query_variants.replacements 中每条规则必须配置 when_any 或 when_all：{config_path}")
    if not replacements:
        raise ValueError(f"query_variants.replacements 中每条规则必须配置 replace：{config_path}")
    return QueryVariantReplacementRule(
        when_any=when_any,
        when_all=when_all,
        replacements=replacements,
        ignore_case=bool(rule_payload.get("ignore_case", False)),
    )


def _load_intent_rule_score_rules(payload: dict, config_path: Path) -> IntentRuleScoreRules:
    """Parse deterministic intent-rule scores from TOML payload.

    调用顺序：启动配置或前置校验 -> _load_intent_rule_score_rules()。
    """

    raw = dict(payload.get("intent_rule_scores") or {})
    required_keys = (
        "strong_faq",
        "knowledge",
        "source_question_shape",
        "direct_faq_shape",
    )
    missing = [key for key in required_keys if key not in raw]
    if missing:
        raise ValueError(f"intent_rule_scores 缺少配置项 {missing}：{config_path}")

    scores = {key: _as_float(raw[key], f"intent_rule_scores.{key}", config_path) for key in required_keys}
    for key, value in scores.items():
        if not 0 < value <= 1:
            raise ValueError(f"intent_rule_scores.{key} 必须在 0 到 1 之间：{config_path}")

    if not (
        scores["strong_faq"]
        < scores["knowledge"]
        < scores["source_question_shape"]
        < scores["direct_faq_shape"]
    ):
        raise ValueError(
            "intent_rule_scores 必须满足 strong_faq < knowledge < "
            f"source_question_shape < direct_faq_shape：{config_path}"
        )

    return IntentRuleScoreRules(
        strong_faq=scores["strong_faq"],
        knowledge=scores["knowledge"],
        source_question_shape=scores["source_question_shape"],
        direct_faq_shape=scores["direct_faq_shape"],
    )


def _load_retrieval_strategy_rules(payload: dict, config_path: Path) -> RetrievalStrategyRules:
    """Parse retrieval strategy guardrails from TOML payload.

    调用顺序：启动配置或前置校验 -> _load_retrieval_strategy_rules()。
    """

    raw = dict(payload.get("retrieval_strategy") or {})
    required_float_keys = (
        "faq_direct_floor",
        "faq_direct_discount",
        "short_query_guard_threshold",
        "low_rule_score_threshold",
        "medium_rule_score_threshold",
        "medium_rule_score_direct_threshold",
        "pricing_direct_threshold",
        "compliance_direct_threshold",
        "low_rule_score_direct_threshold",
    )
    required_int_keys = (
        "follow_up_faq_top_k_min",
        "strong_faq_doc_top_k_min",
        "knowledge_context_top_n_min",
        "guard_context_top_n_min",
        "table_context_top_n_min",
    )
    missing = [key for key in (*required_float_keys, *required_int_keys) if key not in raw]
    if missing:
        raise ValueError(f"retrieval_strategy 缺少配置项 {missing}：{config_path}")

    floats = {key: _as_float(raw[key], f"retrieval_strategy.{key}", config_path) for key in required_float_keys}
    ints = {key: _as_positive_int(raw[key], f"retrieval_strategy.{key}", config_path) for key in required_int_keys}
    if not 0 < floats["faq_direct_discount"] < 1:
        raise ValueError(f"retrieval_strategy.faq_direct_discount 必须在 0 到 1 之间：{config_path}")
    for key in required_float_keys:
        if key == "faq_direct_discount":
            continue
        if not 0 < floats[key] <= 1:
            raise ValueError(f"retrieval_strategy.{key} 必须在 0 到 1 之间：{config_path}")
    if floats["low_rule_score_threshold"] >= floats["medium_rule_score_threshold"]:
        raise ValueError(
            f"retrieval_strategy.low_rule_score_threshold 必须小于 medium_rule_score_threshold：{config_path}"
        )

    return RetrievalStrategyRules(
        faq_direct_floor=floats["faq_direct_floor"],
        faq_direct_discount=floats["faq_direct_discount"],
        short_query_guard_threshold=floats["short_query_guard_threshold"],
        low_rule_score_threshold=floats["low_rule_score_threshold"],
        medium_rule_score_threshold=floats["medium_rule_score_threshold"],
        medium_rule_score_direct_threshold=floats["medium_rule_score_direct_threshold"],
        pricing_direct_threshold=floats["pricing_direct_threshold"],
        compliance_direct_threshold=floats["compliance_direct_threshold"],
        low_rule_score_direct_threshold=floats["low_rule_score_direct_threshold"],
        follow_up_faq_top_k_min=ints["follow_up_faq_top_k_min"],
        strong_faq_doc_top_k_min=ints["strong_faq_doc_top_k_min"],
        knowledge_context_top_n_min=ints["knowledge_context_top_n_min"],
        guard_context_top_n_min=ints["guard_context_top_n_min"],
        table_context_top_n_min=ints["table_context_top_n_min"],
    )


def _as_float(value: object, name: str, config_path: Path) -> float:
    """Parse a float config value with a useful error.

    调用顺序：启动配置或前置校验 -> _as_float()。
    """

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字：{config_path}") from exc


def _as_positive_int(value: object, name: str, config_path: Path) -> int:
    """Parse a positive integer config value with a useful error.

    调用顺序：启动配置或前置校验 -> _as_positive_int()。
    """

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数：{config_path}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} 必须大于 0：{config_path}")
    return parsed


def _clean_tuple(items: object) -> tuple[str, ...]:
    """Return non-empty strings as a tuple.

    调用顺序：启动配置或前置校验 -> _clean_tuple()。
    """

    return tuple(str(item).strip() for item in items or () if str(item).strip())
