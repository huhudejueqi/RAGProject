# prepare_retrieval 完整策略文档

## 概览

`prepare_retrieval` 将用户问题和对话历史转换为一份**检索参数包** `RetrievalPreparation`，
包含检索所需全部参数（意图、过滤条件、改写结果、检索计划、查询变体、提示词模板），
下游 `search_faq` / `search_doc` / `prepare_answer` 直接消费。

内部依次执行 7 个阶段：

```
load_history
  → classify_intent
  → resolve_source_filter
  → rewrite_query_if_needed
  → build_retrieval_plan
  → generate_query_variants
  → select_prompt_profile
```

---

## 阶段 1: load_history

```
输入: session_id
输出: history_messages (list[BaseMessage])
耗时: MySQL 查询
```

从 MySQL `chat_messages` 表加载当前会话的对话历史。
格式为：`[SystemMessage(摘要), HumanMessage, AIMessage, ...]`。

**记忆压缩**（`ChatHistoryStore.refresh_summary_if_needed`）：
- 每轮问答结束后异步触发
- 只对"旧消息"做 LLM 增量摘要，保留最近 N 条原文
- 下次加载时，摘要作为 `SystemMessage("历史摘要：...")`，原文作为 `[HumanMessage, AIMessage]`
- 配置：
  - `history_summary_after_messages`: 消息数达到此阈值才生成摘要
  - `history_recent_messages`: 保留原文的消息数

---

## 阶段 2: classify_intent

```
输入: query, history_messages, scenario
输出: IntentResult
状态: intent_payload
```

**意图分类**，确定用户问题属于哪类检索。

### 2.1 追问检测

```python
if history and (FOLLOW_UP_HINTS.search(query) or len(query) <= 8):
    → FOLLOW_UP, requires_rewrite=True
```

FOLLOW_UP_HINTS 正则：`^(那|这个|那个|它|他们|她们|这些|上面|刚才|继续|还有|费用呢|审批呢|权限呢|发票呢|告警呢|步骤呢)`

| 例子 | 匹配原因 |
|------|----------|
| "那审批呢" | 以"那"开头 |
| "继续" | 精确匹配 |
| "多少钱" | 长度 ≤ 8 |
| "怎么做" | 长度 ≤ 8 |

**必须 history 不为空**，否则不判追问。

### 2.2 领域规则匹配

`_strong_rule_domain_intent` — 四条正则规则，**收集所有候选，取最高分**。

| 规则 | 分值 | 匹配条件 | 例子 |
|------|------|---------|------|
| FAQ关键词 | 0.82 | `FAQ_HINTS` 命中：费用、发票、报错、权限... | "报销多少" |
| 标准问法 | 0.85 | 有 source + 长度≤32 + `FAQ_QUESTION_SHAPE_HINTS` 命中：怎么办、需要哪些... | "入职需要哪些材料" |
| 直接问法 | 0.86 | 有 source + 长度≤36 + `DIRECT_FAQ_SHAPE_HINTS` 命中：是什么、可以吗... | "流程是什么" |
| 知识查询 | 0.84 | `KNOWLEDGE_HINTS` 命中 + (有source 或 长度≤24)：流程、制度、文档... | "查看薪酬制度" |

**不短路优先，而是全部收集后取最高分**。防止低分规则因为写在前面就抢跑。

相关正则定义：

| 正则 | 定义位置 | 内容 |
|------|---------|------|
| `FAQ_HINTS` | classifier.py:34 | 费用、价格、安装、报错、发票、权限... |
| `KNOWLEDGE_HINTS` | classifier.py:35 | 知识库、文档、流程、制度、SOP、配置... |
| `FAQ_QUESTION_SHAPE_HINTS` | classifier.py:36 | 怎么办、需要什么、有哪些、为什么... |
| `DIRECT_FAQ_SHAPE_HINTS` | classifier.py:37 | 是什么、可以吗、能不能、怎么处理... |

### 2.3 兜底

所有规则未命中 → `KNOWLEDGE_QUERY(0.6)`，标记 `default_knowledge`。

### 2.4 规则-模型融合

`apply_intent_decision_gateway` 将规则结果与 BERT 模型结果融合：

| 优先级 | 条件 | 决策 |
|--------|------|------|
| 1 | 模型判追问但无历史 | 规则保持（模型误判） |
| 2 | 模型置信度 < 0.55 | 规则保持（模型不可信） |
| 3 | 规则与模型一致 | 取高分 + 0.03 加成 |
| 4 | 规则是 default_knowledge | 采纳模型 |
| 5 | 冲突 | 规则保持但降至 0.68 |

BERT 模型只有三个标签：`FAQ_QUERY` / `KNOWLEDGE_QUERY` / `FOLLOW_UP`。

`suggested_source` 通过 `infer_source` 从 `source_patterns` 正则匹配得出。

---

## 阶段 3: resolve_source_filter

```
输入: source_filter(前端), suggested_source(意图), scenario
输出: effective_source_filter(str|None)
```

确定最终用于 Milvus 过滤的 source 值：

```
① 前端显式选择 → 用前端的
② 前端未选 + 意图推断有效 → 用推断的（需在白名单内）
③ 都没有 → None（不限分类）
```

| source_filter(前端) | suggested_source(推断) | 结果 |
|---------------------|----------------------|------|
| "hr" | "it" | **"hr"** |
| None | "it" | **"it"** |
| None | None | **None** |
| None | "legal"（不在白名单） | **None** |

---

## 阶段 4: rewrite_query_if_needed

```
输入: query, history_messages, requires_rewrite
输出: rewritten_query (str)
```

仅当 `intent.requires_rewrite=True`（即 FOLLOW_UP）时执行。
把依赖上下文的追问改写成独立可检索的问题。

| 原始追问 | 历史 | 改写后 |
|---------|------|--------|
| "那审批呢" | "入职流程是什么" | "入职审批流程是什么" |
| "费用呢" | "报销需要哪些材料" | "报销费用规定是什么" |
| "然后" | "VPN怎么配置" | "VPN配置后续步骤" |

未命中追问时，`rewritten_query = query` 不变。

---

## 阶段 5: build_retrieval_plan

```
输入: rewritten_query, intent
输出: RetrievalPlan
状态: plan
```

**核心决策层**。将意图转为具体检索参数。内部 5 层规则**逐层覆盖**：

### 5.1 基础参数

```python
_base_params(settings, is_short)
```

| 参数 | 默认值 | 短问题时 |
|------|--------|---------|
| faq_top_k | 20 | 30（短问题多搜FAQ） |
| doc_top_k | 20 | 20 |
| direct_threshold | 0.72 | 0.72 |
| final_context_top_n | 4 | 4 |
| run_faq | True | True |
| run_doc | True | True |

### 5.2 意图分支补丁

`_intent_rules(settings)` 返回的 `PlanPatch`：

| 意图 | reason | 改动 | 效果 |
|------|--------|------|------|
| FAQ_QUERY | faq_first | doc_top_k=10(减半), direct_threshold=0.64(降低) | FAQ优先、文档降噪、易直出 |
| KNOWLEDGE_QUERY | knowledge_doc_enriched | doc_top_k_min=24, final_context_top_n_min=5 | 扩大文档召回、多取上下文 |
| FOLLOW_UP | history_aware_follow_up | faq_top_k_min=24, doc_top_k_min=24, final_context_top_n_min=5, direct_threshold_min=0.82 | 全面扩大召回、最保守直出 |

### 5.3 短问题保护

条件：`is_short=True`（长度 ≤ 20）且非追问

```python
PlanPatch(
    reason="short_query_guard",
    doc_top_k_max=max(12, 4*2)=12,        # 文档最多 12 条
    direct_threshold_min=0.78,             # FAQ 直出门槛提高到 0.78
)
```

短问题歧义大（"怎么做"、"费用呢"），限制文档量避免噪音、提高门槛防止误直出。

### 5.4 规则分数保护

`_rule_score_guard(settings, rules, confidence)` — 根据 `intent.confidence`（规则+模型融合后的最终分数）：

| 分数 | 补丁 | 含义 |
|------|------|------|
| < 0.70 | low_rule_score_guard | 低分兜底，最大保守措施 |
| < 0.82 | rule_score_guard | 中度保守 |
| ≥ 0.82 | None | 不保护 |

低分补丁详情：

```python
# confidence < 0.70
doc_top_k_min=24,                          # 扩大文档召回
final_context_top_n_min=6,                 # 至少6条上下文
direct_threshold_min=max(0.72, 0.86)=0.86, # 直出阈值提到最高
faq_direct_exact_only=True,                # 只允许精确匹配直出
```

```python
# 0.70 ≤ confidence < 0.82
doc_top_k_min=20,                          # 恢复默认文档量
direct_threshold_min=max(0.72, 0.82)=0.82, # 阈值提到 0.82
```

### 5.5 风险类别保护

`infer_question_category(query)` 返回风险类型，各类型对应不同补丁：

| 类别 | 关键字 | 补丁 | 原因 |
|------|--------|------|------|
| pricing | 费用、价格、金额、单价... | doc扩到24条、阈值提到0.84、上下文至少6条 | 金额错误 → 经济损失 |
| compliance | 合规、法规、监管、条款... | doc扩到24条、阈值提到0.86、上下文至少6条 | 合规错误 → 监管处罚 |
| troubleshooting | 报错、失败、故障、排查... | doc扩到24条、阈值降0.70、上下文至少6条 | 需要充分的步骤化上下文 |
| summary | 总结、汇总、归纳... | doc扩到24条、上下文至少6条 | 信息量大，精度要求相对宽松 |
| default | 其他 | 无补丁 | 常规策略 |

### 5.6 表格偏好保护

`is_table_query(query)` 判断是否为表格类查询（清单、台账、字段、sheet...）。

命中后：

```python
PlanPatch(
    reason="table_row_preferred",
    doc_top_k_min=24,          # 扩大文档候选
    final_context_top_n_min=7, # 至少 7 条上下文
    faq_direct_exact_only=True, # 禁用模糊 FAQ 直出
)
```

表格数据不同行列语义相近但内容完全不同（如"验收项清单"vs"付款节点清单"），
模糊 FAQ 直出极容易给错答案。

### 5.7 `_apply_patch` 机制

每条规则是一个 `PlanPatch`，通过 `_apply_patch(params, patch)` 应用到参数：

| 字段 | 行为 | 举例 |
|------|------|------|
| faq_top_k | 直接覆盖 | 设为 12 |
| faq_top_k_min | 取 max（不能低于下限） | 当前 8，补丁要求≥10 → 10 |
| doc_top_k_max | 取 min（不能超过上限） | 当前 20，补丁要求≤12 → 12 |
| direct_threshold_min | 取 max（不能低于下限） | 当前 0.72，补丁要求≥0.78 → 0.78 |
| reason | 追加策略标签 | `balanced_retrieval_short_query_guard` |

多层规则叠加时，`_min`/`_max` 确保后层不会突破前层的约束。

### 5.8 参数优先级汇总

```
基础参数 ← 意图分支 ← 短问题保护 ← 规则分保护 ← 风险类别 ← 表格偏好
  (20)      (doc=10)   (doc_max=12)  (doc_min=24)                (doc_min=24)
                                                                    ↓
最终 doc_top_k = min(max(10, 24), 12) = 12
```

---

## 阶段 6: generate_query_variants

```
输入: rewritten_query, plan.use_query_variants, intent
输出: query_variants (list[str])
```

为问题生成同义检索表达，提升召回率。**仅知识查询和追问启用**（FAQ 标准问题固定不变，不需要）。

### 6.1 规则替换层

`_heuristic_variants()` — 零成本本地替换：

| 原词 | 替换为 |
|------|--------|
| 流程 | SOP、处理步骤、办理流程 |
| 发票 | 开票、账单 |
| webhook | 回调 |
| 告警 | 报警、异常 |
| 资料 | 材料、记录 |
| 安装、失败、报错 | 相互替换 |
| 能不能 | 是否可以 |
| 可以吗 | 是否可以 |

### 6.2 LLM 兜底层

规则生成足够变体则直接返回；不够才调 LLM 生成。

### 6.3 启用条件

```python
use_query_variants = intent.intent in {"KNOWLEDGE_QUERY", "FOLLOW_UP"}
```

| 意图 | 启用 | 原因 |
|------|------|------|
| FAQ_QUERY | ❌ | FAQ 标准问题固定不变，变体增加误匹配风险 |
| KNOWLEDGE_QUERY | ✅ | 文档内容措辞不固定，变体提升召回 |
| FOLLOW_UP | ✅ | 追问短，变体能覆盖更多可能 |

---

## 阶段 7: select_prompt_profile

```
输入: intent.intent, scenario, rewritten_query
输出: prompt_profile (含 system_template + user_template)
```

根据意图和风险类别选择回答模板档位。不同模板包含不同约束和免责声明：

| 用途 | 模板特点 |
|------|---------|
| 常规回答 | 通用约束 |
| 费用类 | 金额准确性强调 + 免责声明 |
| 合规类 | 法规引用要求 + 免责声明 |
| 排障类 | 步骤化操作指引 |
| 总结类 | 允许概括性表述 |

---

## 输出: RetrievalPreparation

```python
@dataclass
class RetrievalPreparation:
    history_messages: list[BaseMessage]    # 阶段1 输出
    intent: IntentResult                   # 阶段2 输出
    effective_source_filter: str | None    # 阶段3 输出
    rewritten_query: str                   # 阶段4 输出
    plan: RetrievalPlan                    # 阶段5 输出
    query_variants: list[str]              # 阶段6 输出
    prompt_profile: PromptProfile          # 阶段7 输出
```

下游 `search_faq` / `search_doc` / `prepare_answer` 直接消费此包，不再重复计算。
