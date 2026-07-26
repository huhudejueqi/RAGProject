"""
================================================================================
完整 RAG 查询闭环 Demo：Milvus 混合召回 → Rerank 重排序 → 本地大模型回答
================================================================================

本文件演示了一个完整的企业级 RAG（检索增强生成，Retrieval-Augmented Generation）流程，
包含三个核心环节：

1. 混合召回（Hybrid Search）
   - 使用 BGE-M3 嵌入模型分别对文档的"标题"和"正文"两个字段生成 dense 向量
   - 查询时同时对 title_vector 和 content_vector 两个向量字段执行 ANN 近似最近邻搜索
   - 通过 WeightedRanker 加权融合两个字段的召回结果
   - 标题权重 0.35，正文权重 0.65（正文包含更丰富的语义信息，查全率更重要）

2. 重排序（Rerank）
   - 使用 bge-reranker-large 交叉编码器（Cross-Encoder）对召回结果进行精细排序
   - 将 query 与每篇候选文档拼接后送入模型，逐一计算深度语义相关性分数
   - 交叉编码器比双塔模型的余弦相似度更精准（能捕捉 query-doc 之间的细粒度交互）
   - 采用"粗筛 → 精排"两阶段策略：向量召回快速海选，Rerank 精准排序

3. 大模型生成（LLM Generation）
   - 将重排序后的文档作为上下文，与用户问题拼装为结构化 Prompt
   - 调用ChatOpenAI千问大语言模型，基于资料生成最终答案
   - 支持 --no-llm 参数：仅输出 Prompt 而不调用模型，方便调试

   xxxx.py --n "ssssd" -n  weww

技术栈：
  - 向量模型：  BGE-M3（BAAI 开源，支持 dense / sparse / colbert 三种向量）
  - 重排序模型：bge-reranker-large（BAAI 开源，交叉编码器架构）
  - 大语言模型：ChatGLM-6B（清华大学开源，60 亿参数中英双语对话模型）
  - 向量数据库：Milvus（支持混合搜索 + 加权融合排序）

运行示例：
  python demo08_hybrid_search_rerank_rag.py --query "入职需要提交哪些材料？" --rebuild
  python demo08_hybrid_search_rerank_rag.py --query "跨境申报要检查什么？" --no-llm
"""

from __future__ import annotations  # 延迟类型注解求值，支持前向引用（PEP 563）

import argparse  # 命令行参数解析
import importlib.util  # 动态加载模块（用于修复 ChatGLM tokenizer 兼容性）
import os
from functools import wraps
from pathlib import Path  # 面向对象的文件路径处理

import torch  # PyTorch 深度学习框架
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pymilvus import AnnSearchRequest, DataType, WeightedRanker  # Milvus Python SDK
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer  # HuggingFace Transformers

from milvus_common import connect_client  # 公共模块：创建 Milvus 客户端连接
load_dotenv()
print(os.environ.get("LLM_MODEL"))


def patch_transformers_dtype_compat():
    """把 FlagEmbedding 1.4.x 传入的 dtype 参数映射为 transformers 可识别的 torch_dtype。"""

    def patch_from_pretrained(model_cls):
        original = model_cls.from_pretrained
        if getattr(original, "_dtype_compat_patched", False):
            return

        @classmethod
        @wraps(original)
        def wrapped(cls, *args, **kwargs):
            if "dtype" in kwargs and "torch_dtype" not in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            return original(*args, **kwargs)

        wrapped._dtype_compat_patched = True  # type: ignore[attr-defined]
        model_cls.from_pretrained = wrapped

    patch_from_pretrained(AutoModel)
    patch_from_pretrained(AutoModelForSequenceClassification)


patch_transformers_dtype_compat()

# ============================================================================
# 全局常量配置
# ============================================================================

# Milvus Collection 名称 —— 用于存储本次 demo 的文档向量和元数据
COLLECTION_NAME = "lecture04_rag_query_demo"

# 本地预下载模型的根目录
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# BGE-M3 嵌入模型路径 —— 将文本转换为 dense 稠密向量（1024 维）
# 模型来源：BAAI/bge-m3，支持中英文，可同时输出 dense / sparse / colbert 三种向量
EMBED_MODEL_PATH = MODELS_DIR / "bge-m3"

# bge-reranker-large 重排序模型路径 —— 交叉编码器，对召回结果进行精细重排序
# 模型来源：BAAI/bge-reranker-large，基于 xlm-roberta-large 架构
RERANK_MODEL_PATH = MODELS_DIR / "bge-reranker-large"

# ChatGLM-6B 大语言模型路径 —— 最终答案生成
# 模型来源：THUDM/chatglm-6b，清华大学开源的中英双语对话模型
LLM_MODEL_PATH = MODELS_DIR / "ChatGLM-6B"


# ============================================================================
# 示例文档数据集（模拟企业知识库）
# ============================================================================
# 覆盖 HR（人力资源）、Finance（财务）、Trade（贸易合规）三个业务领域
# 每条文档字段说明：
#   pk:       主键（Primary Key），Milvus 实体唯一标识，命名格式 "{类别}_{序号}"
#   title:    文档标题 —— 用于标题向量检索（精准关键词匹配）
#   content:  文档正文 —— 用于正文向量检索（语义相似匹配）
#   category: 业务类别标签，可用于后续的元数据过滤
DOCS = [
    {
        "pk": "hr_001",
        "title": "员工入职材料清单",
        "content": "员工入职前需要提交身份证、学历证明、银行卡信息、体检报告，并完成劳动合同签署。",
        "category": "hr",
    },
    {
        "pk": "hr_002",
        "title": "员工离职交接流程",
        "content": "离职交接需要完成资产归还、账号注销、权限回收、工作交接确认，并由直属主管审批。",
        "category": "hr",
    },
    {
        "pk": "finance_001",
        "title": "费用报销凭证要求",
        "content": "费用报销需要提供合规发票、审批单、付款凭证和费用说明，超预算项目还需要预算追加审批。",
        "category": "finance",
    },
    {
        "pk": "trade_001",
        "title": "跨境申报资料检查",
        "content": "跨境申报前需要检查 HS 编码、原产地证明、装箱单、商业发票和许可证要求。",
        "category": "trade",
    },
    {
        "pk": "trade_002",
        "title": "制裁名单命中处理",
        "content": "交易对象命中制裁名单后，应暂停交易，保存筛查记录，并提交合规部门复核。",
        "category": "trade",
    },
]


# ============================================================================
# 向量编码
# ============================================================================

def encode_texts(embed_model, texts):
    """
    使用 BGE-M3 模型将文本列表编码为 dense 稠密向量。

    BGE-M3 是 BAAI 开源的多语言嵌入模型，能够同时输出三种向量表示：
      - dense_vecs:    稠密向量（1024 维浮点数），擅长捕捉深层语义相似度
      - sparse_vecs:   稀疏向量（词袋类权重），适合精确关键词匹配
      - colbert_vecs:  ColBERT 多向量（token 级），适合细粒度交互匹配

    本 demo 仅使用 dense 向量 —— 在精度和效率之间取得平衡。

    参数：
        embed_model: BGE-M3 模型实例（FlagEmbedding.BGEM3FlagModel）
        texts:       待编码文本列表，如 ["文本1", "文本2", ...]

    返回：
        List[List[float]]: 二维浮点数列表，形状为 [len(texts), 1024]
                           每个内层列表是对应文本的 dense 向量
    """
    # 调用 BGE-M3 的 encode 方法，仅提取 dense 稠密向量
    result = embed_model.encode(
        texts,
        return_dense=True,          # 返回稠密向量（核心输出）
        return_sparse=False,        # 不需要稀疏向量
        return_colbert_vecs=False,  # 不需要 ColBERT 向量
    )
    vectors = result["dense_vecs"]

    # numpy 数组 → Python 原生列表（Milvus SDK 的要求）
    # hasattr 检查确保兼容性：numpy 数组有 .tolist()，而 Python list 没有
    return vectors.tolist() if hasattr(vectors, "tolist") else vectors


# ============================================================================
# Milvus Collection 准备
# ============================================================================

def prepare_milvus(client, embed_model, rebuild=False):
    """
    初始化 Milvus Collection：创建 Schema、构建索引、写入文档、加载到内存。

    Collection Schema 设计：
    ┌────────────────┬──────────────┬──────────────────────────────────┐
    │ 字段名         │ 数据类型     │ 用途                             │
    ├────────────────┼──────────────┼──────────────────────────────────┤
    │ pk             │ VARCHAR(64)  │ 主键，文档唯一标识                │
    │ title          │ VARCHAR(256) │ 文档标题（原文，用于展示）         │
    │ content        │ VARCHAR(2048)│ 文档正文（原文，用于生成答案）     │
    │ category       │ VARCHAR(64)  │ 业务类别标签（hr/finance/trade）  │
    │ title_vector   │ FLOAT_VECTOR │ 标题的 dense 向量（混合召回通道1） │
    │ content_vector │ FLOAT_VECTOR │ 正文的 dense 向量（混合召回通道2） │
    └────────────────┴──────────────┴──────────────────────────────────┘

    向量索引采用 HNSW（Hierarchical Navigable Small World，分层可导航小世界图）：
      - 当前最主流的 ANN（近似最近邻）索引之一
      - M=16: 图中每个节点最多连接 16 个邻居（值越大召回率越高，但索引越大）
      - efConstruction=100: 构建索引时的搜索宽度（值越大索引质量越高，构建越慢）
      - 度量方式: COSINE（余弦相似度，值域 [-1, 1]，1 表示完全相同）

    参数：
        client:       Milvus 客户端连接对象
        embed_model:  BGE-M3 模型实例，用于生成向量
        rebuild:      是否重建 Collection（True = 删除已有数据后重建）
    """
    # ----- 重建逻辑 -----
    # 如果指定 --rebuild 且 Collection 已存在，先删除旧的再重新创建
    # 保证每次重建都使用最新的数据结构和向量
    if rebuild and client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    # ----- 首次创建 Collection + Schema -----
    if not client.has_collection(COLLECTION_NAME):
        # 用一条测试文本探测 BGE-M3 的向量维度（通常为 1024）
        dim = len(encode_texts(embed_model, ["维度测试"])[0])

        # 创建 Schema
        # auto_id=False: 由用户显式指定主键值，而非 Milvus 自动生成
        # enable_dynamic_field=False: 禁用动态字段，所有字段必须预先定义（更严格的 Schema 管控）
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

        # 元数据字段 —— 存储文档原文，用于检索后展示和生成答案
        schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("title", DataType.VARCHAR, max_length=256)
        schema.add_field("content", DataType.VARCHAR, max_length=2048)
        schema.add_field("category", DataType.VARCHAR, max_length=64)

        # 向量字段 —— 分别对标题和正文建立独立的向量列
        # 这样混合搜索时可以分别检索两个字段，再加权融合结果
        schema.add_field("title_vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("content_vector", DataType.FLOAT_VECTOR, dim=dim)

        # 创建 Collection
        # consistency_level="Session": 会话级一致性 —— 同一会话内写入立即可见，适合 demo 场景
        client.create_collection(COLLECTION_NAME, schema=schema, consistency_level="Session")

        # ----- 创建向量索引 -----
        # 为 title_vector 和 content_vector 两个字段分别创建 HNSW 索引
        index_params = client.prepare_index_params()
        for field in ["title_vector", "content_vector"]:
            index_params.add_index(
                field,
                index_name=f"{field}_hnsw",    # 索引命名：title_vector_hnsw / content_vector_hnsw
                index_type="HNSW",              # HNSW 图索引
                metric_type="COSINE",           # 余弦相似度
                params={"M": 16, "efConstruction": 100},
            )
        client.create_index(COLLECTION_NAME, index_params)

    # ----- 编码并写入文档 -----
    # 分别对标题文本和正文文本生成向量（都使用 BGE-M3）
    title_vectors = encode_texts(embed_model, [doc["title"] for doc in DOCS])
    content_vectors = encode_texts(embed_model, [doc["content"] for doc in DOCS])

    # 组装实体行：将原始元数据和两个向量字段合并为一条记录
    rows = []
    for doc, title_vector, content_vector in zip(DOCS, title_vectors, content_vectors):
        row = dict(doc)                    # 浅拷贝原始文档 {pk, title, content, category}
        row["title_vector"] = title_vector     # 附加标题向量
        row["content_vector"] = content_vector # 附加正文向量
        rows.append(row)

    # 批量 upsert（存在则更新，不存在则插入）
    client.upsert(COLLECTION_NAME, rows)

    # ----- 持久化与加载 -----
    # flush: 将内存中的增量数据刷写到持久化存储，防止数据丢失
    client.flush(COLLECTION_NAME)
    # load_collection: 将 Collection 的索引和数据加载到内存，使搜索功能可用
    # 未 load 的 Collection 无法执行 search / query 操作
    client.load_collection(COLLECTION_NAME)


# ============================================================================
# 混合搜索（Hybrid Search）
# ============================================================================

def hybrid_search(client, embed_model, query, top_k=5):
    """
    对标题向量和正文向量同时执行 ANN 搜索，通过加权融合得到最终排序。

    混合搜索的核心价值：
      同一查询分别检索不同的向量字段，再将多个召回通路的结果融合。
      不同字段捕捉不同类型的相关性：
        - 标题向量匹配 → 精准关键词匹配（如"入职材料清单" vs "入职"）
        - 正文向量匹配 → 宽泛语义匹配（如"提交哪些文件" vs "需要提交身份证…"）

    加权融合公式（由 WeightedRanker 在 Milvus 内部完成）：
      final_score = 0.35 × cosine(query, title_vector) + 0.65 × cosine(query, content_vector)
      （Milvus 会先对各通路分数做归一化，再加权求和）
      正文权重 0.65 > 标题权重 0.35，因为正文语义信息更丰富

    参数：
        client:       Milvus 客户端
        embed_model:  BGE-M3 模型实例
        query:        用户查询文本
        top_k:        最终返回的结果数（默认 5）

    返回：
        List[dict]: 融合排序后的候选文档列表，每条包含：
                    pk, title, content, category, recall_score
    """
    # 将查询文本编码为向量（同一个查询向量用于两个字段的搜索）
    query_vector = encode_texts(embed_model, [query])[0]

    # ----- 构建标题字段的 ANN 搜索请求 -----
    # AnnSearchRequest 封装单次向量搜索的完整参数
    title_req = AnnSearchRequest(
        data=[query_vector],             # 查询向量（必须放在列表中）
        anns_field="title_vector",       # 目标向量字段：标题向量
        param={
            "metric_type": "COSINE",     # 余弦相似度
            "params": {"ef": 64},        # HNSW 搜索参数 ef：搜索时的候选池大小
                                         # ef 越大召回率越高，但搜索耗时越长
        },
        limit=top_k,                     # 该通路返回的最大候选数
    )

    # ----- 构建正文字段的 ANN 搜索请求 -----
    content_req = AnnSearchRequest(
        data=[query_vector],
        anns_field="content_vector",     # 目标向量字段：正文向量
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
    )

    # ----- 执行混合搜索 -----
    # WeightedRanker(0.35, 0.65):
    #   两个权重分别对应 reqs 列表中的 title_req 和 content_req
    #   Milvus 内部处理流程：
    #     1) 分别执行两个 ANN 搜索
    #     2) 对每个通路的分数做归一化（使两个通路的分数可比较）
    #     3) 按权重加权求和得到最终分数
    #     4) 按最终分数降序排列，返回 top_k 条
    results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[title_req, content_req],              # 两个搜索请求
        ranker=WeightedRanker(0.35, 0.65),          # 加权融合策略
        limit=top_k,
        output_fields=["pk", "title", "content", "category"],  # 返回的元数据字段
    )

    # ----- 解析结果 -----
    # results[0] 是融合后的 TopK 结果列表
    # 每条结果包含:
    #   id:      实体的内部 ID
    #   distance: 加权融合后的最终分数（越高越相关）
    #   entity:   请求的 output_fields 字段值字典
    hits = []
    for hit in results[0]:
        entity = hit["entity"]
        hits.append(
            {
                "pk": entity["pk"],
                "title": entity["title"],
                "content": entity["content"],
                "category": entity["category"],
                "recall_score": float(hit["distance"]),  # 加权融合后的最终分数
            }
        )
    return hits


# ============================================================================
# 重排序（Rerank）
# ============================================================================

def rerank(query, hits, top_k=3):
    """
    使用 bge-reranker-large 交叉编码器对混合召回结果进行精细重排序。

    为什么需要 Rerank？
      ┌────────────┬──────────────────────┬──────────────────────┐
      │            │ 向量召回（双塔模型）   │ Rerank（交叉编码器）  │
      ├────────────┼──────────────────────┼──────────────────────┤
      │ 编码方式   │ query 和 doc 独立编码  │ query 和 doc 拼接编码 │
      │ 速度       │ 快（可预计算 doc 向量）│ 慢（每次需重新计算）  │
      │ 精度       │ 一般                   │ 高（捕捉细粒度交互） │
      │ 适用阶段   │ 海量粗筛（召回）       │ 少量精排（重排序）   │
      └────────────┴──────────────────────┴──────────────────────┘

    采用"粗筛 → 精排"两阶段策略：
      - 第一阶段：向量召回从全量文档中快速筛选 TopK 候选（本 demo 为 5 条）
      - 第二阶段：交叉编码器对少量候选精细打分，选出最相关的 TopK（本 demo 为 3 条）

    bge-reranker-large 原理：
      - 架构：基于 xlm-roberta-large，在顶部加了分类头
      - 输入：(query, document) 文本对
      - 输出：一个标量 logit，表示 query-document 的相关程度
      - 与余弦相似度不同，交叉编码器的 self-attention 能让 query 和 doc 的
        每个 token 相互 attend，从而捕获深层的语义交互

    参数：
        query:  用户查询文本
        hits:   混合召回返回的候选文档列表
        top_k:  重排序后保留的最终文档数（默认 3）

    返回：
        List[dict]: 重排序后的文档列表，按 rerank_score 降序排列
                    每条新增 rerank_score 字段
    """
    # ----- 加载重排序模型 -----
    # bge-reranker 使用自定义的 sentencepiece 分词器
    # vocab_file 显式指定了 sentencepiece 模型文件路径
    tokenizer = AutoTokenizer.from_pretrained(
        str(RERANK_MODEL_PATH),
        vocab_file=str(RERANK_MODEL_PATH / "sentencepiece.bpe.model"),
        use_fast=False,  # 使用 Python 实现的分词器（比 Rust 版本兼容性更好）
    )

    # AutoModelForSequenceClassification:
    #   自动加载序列分类模型 —— 在预训练语言模型顶部加一个线性分类头
    #   输出的 logits 就是文档与查询的相关性分数
    model = AutoModelForSequenceClassification.from_pretrained(str(RERANK_MODEL_PATH))

    # model.eval(): 切换到评估模式
    #   - 冻结 Dropout 层（推理时不做随机丢弃，保证结果确定性）
    #   - 冻结 BatchNorm 层（使用训练时积累的统计量，而非批次统计量）
    model.eval()

    # ----- 构造 (查询, 文档) 文本对 -----
    # 将标题和正文用换行符拼接为完整文档文本
    # 格式示例: ("入职需要什么", "员工入职材料清单\n员工入职前需要提交身份证...")
    pairs = [[query, f"{hit['title']}\n{hit['content']}"] for hit in hits]

    # ----- 批量推理：计算相关性分数 -----
    # padding=True:     将同一批次中的序列填充到等长（填充到批次内最长序列）
    # truncation=True:  超过 max_length 的序列从尾部截断
    # max_length=512:   限制输入长度为 512 token（平衡精度和显存）
    # return_tensors="pt": 返回 PyTorch 张量
    inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # torch.no_grad(): 上下文管理器，禁用自动求导
    #   - 推理阶段不需要梯度计算，禁用以节省显存并加速
    with torch.no_grad():
        # model(**inputs).logits:      形状 [batch_size, 1] 的原始分数张量
        # .squeeze(-1):                移除最后一维 → 形状变为 [batch_size]
        # .tolist():                   转为 Python 浮点数列表
        scores = model(**inputs).logits.squeeze(-1).tolist()

    # 处理边界情况：仅 1 条结果时 tolist() 返回单个 float 而非列表
    if isinstance(scores, float):
        scores = [scores]

    # 将 rerank 分数附加到每条结果上
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)

    # 按 rerank_score 降序排列，截断至 top_k 条
    # 分数越高的文档与查询越相关，排在最前面
    return sorted(hits, key=lambda item: item["rerank_score"], reverse=True)[:top_k]


# ============================================================================
# Prompt 构建
# ============================================================================

def build_prompt(query, hits):
    """
    将重排序后的文档拼接为结构化 Prompt，供大模型生成答案。

    Prompt 工程设计要点：
      1. 角色设定：「企业知识库问答助手」—— 让模型进入专业问答模式
      2. 约束条件：「只根据资料回答问题」—— 防止模型利用自身知识编造答案（幻觉）
      3. 兜底策略：「如果资料不足，请说明没有依据」—— 处理知识库覆盖不全的边界情况
      4. 结构化上下文：每条资料编号 + 标题 + 内容 —— 便于模型定位和引用
      5. 输出控制：「请给出简洁答案」—— 限制回答长度，避免冗长

    参数：
        query: 用户原始问题
        hits:  重排序后的 TopK 文档列表

    返回：
        str: 完整的 Prompt 字符串，可直接送入大模型
    """
    # 将多篇文档用双换行分隔拼接为上下文段落
    # 编号从 1 开始（enumerate start=1），方便模型在回答中引用："根据资料1..."
    context = "\n\n".join(
        f"[资料{i}]\n标题：{hit['title']}\n内容：{hit['content']}"
        for i, hit in enumerate(hits, start=1)
    )

    # 构造完整 Prompt：系统指令 + 上下文资料 + 用户问题 + 输出要求
    return f"""你是企业知识库问答助手。请只根据资料回答问题；如果资料不足，请说明没有依据。

{context}

用户问题：{query}

请给出简洁答案："""


# ============================================================================
# ChatGLM Tokenizer 兼容性修复
# ============================================================================

def load_chatglm_tokenizer(model_path):
    """
    修复旧版 ChatGLM tokenizer 在新版 transformers 库下的兼容性问题。

    问题背景：
      ChatGLM-6B 发布时的 tokenizer 基于较老版本的 transformers API。
      随着 transformers 库升级，部分接口发生变化（如 vocab_size 从方法变为属性、
      _pad 签名新增 padding_side 参数等），导致旧代码无法直接在新版 transformers 上运行。
      本函数通过 Monkey Patch 动态修补这些不兼容的接口。

    修复内容：
      1. vocab_size:   改为 property，委托给底层的 sp_tokenizer
      2. get_vocab:    提供词表映射（id ↔ token）
      3. _pad:         过滤新版 transformers 传入的 padding_side 参数，兼容旧签名

    参数：
        model_path: ChatGLM-6B 模型目录的 Path 对象

    返回：
        ChatGLMTokenizer: 修复后的分词器实例
    """
    # 动态加载 ChatGLM 源码中的 tokenization_chatglm.py 模块
    # 不使用常规 import，因为该文件不在 Python path 中
    spec = importlib.util.spec_from_file_location(
        "local_chatglm_tokenizer",
        model_path / "tokenization_chatglm.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # ----- 补丁 1: vocab_size 属性 -----
    # 新版 transformers 期望 vocab_size 是属性（property），而旧版是方法
    # 转到 sp_tokenizer（底层的 sentencepiece 分词器）获取真实的词表大小
    def vocab_size(self):
        return self.sp_tokenizer.num_tokens if hasattr(self, "sp_tokenizer") else 0

    # ----- 补丁 2: get_vocab 方法 -----
    # 返回完整的词表映射 {token_string: token_id}
    # 包含基础词表 + 后添加的特殊 token（added_tokens_encoder）
    def get_vocab(self):
        if not hasattr(self, "sp_tokenizer"):
            return {}
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # ----- 补丁 3: _pad 方法兼容 -----
    # 新版 transformers 在调用 _pad 时会传入 padding_side 参数
    # 旧版 ChatGLM tokenizer 的 _pad 方法不接受此参数 → 过滤掉
    old_pad = module.ChatGLMTokenizer._pad  # 保存原始的 _pad 方法

    def pad(self, *args, **kwargs):
        kwargs.pop("padding_side", None)  # 移除不兼容的参数
        return old_pad(self, *args, **kwargs)

    # 应用 Monkey Patch —— 将修复后的方法注入到 ChatGLMTokenizer 类上
    module.ChatGLMTokenizer.vocab_size = property(vocab_size)
    module.ChatGLMTokenizer.get_vocab = get_vocab
    module.ChatGLMTokenizer._pad = pad

    # 使用修复后的类创建分词器实例
    # ice_text.model: ChatGLM 使用的 sentencepiece 模型文件
    # num_image_tokens=0: 本 demo 不涉及图像 token
    return module.ChatGLMTokenizer(str(model_path / "ice_text.model"), num_image_tokens=0)


# ============================================================================
# 大模型调用
# ============================================================================

def ask_llm(prompt):
    """
    调用本地 ChatGLM-6B 大语言模型，基于检索资料生成最终答案。

    ChatGLM-6B 简介：
      - 清华大学 KEG 实验室与智谱 AI 联合开源
      - 62 亿参数（6.2B），基于 General Language Model（GLM）架构
      - 支持中英双语对话，在中文任务上表现优异
      - 通过 model.chat() 接口使用，内部维护多轮对话历史

    参数：
        prompt: 构造好的 Prompt 字符串（包含上下文资料 + 用户问题 + 指令）

    返回：
        str: 模型生成的文本回答
    """
    # # ----- 加载分词器（经过兼容性修复）-----
    # tokenizer = load_chatglm_tokenizer(LLM_MODEL_PATH)
    #
    # # ----- 加载模型 -----
    # # trust_remote_code=True: ChatGLM 使用了自定义的 modeling 代码（不在 transformers 官方支持中）
    # #                        必须开启此选项才能加载模型仓库中的 modeling_chatglm.py
    # # .float(): 将模型参数转为 FP32（float32）精度
    # #           ChatGLM-6B 原生为 FP16，转 FP32 在纯 CPU 推理时更稳定，避免精度溢出
    # model = AutoModel.from_pretrained(str(LLM_MODEL_PATH), trust_remote_code=True).float()
    #
    # # ----- 兼容性修复：确保 config 中有 num_hidden_layers 属性 -----
    # # 某些版本的 ChatGLM config 使用 num_layers 而非 num_hidden_layers
    # # 新版 transformers 期望后者 —— 此处做兼容映射
    # if not hasattr(model.config, "num_hidden_layers") and hasattr(model.config, "num_layers"):
    #     model.config.num_hidden_layers = model.config.num_layers
    #
    # # 切换到评估模式
    # model.eval()
    #
    # # ----- 调用 ChatGLM 对话接口 -----
    # # model.chat() 是 ChatGLM 特有的高级对话 API，内部封装了：
    # #   - tokenizer 编码
    # #   - 模型前向推理
    # #   - 解码 + 响应格式化
    # # 参数说明：
    # #   tokenizer:   分词器
    # #   prompt:      输入文本（支持带上下文的多轮格式）
    # #   history:     对话历史列表，[] 表示新对话
    # #   max_length:  生成的最大 token 数（512，足够覆盖大部分答案）
    # #   do_sample:   是否使用采样生成
    # #                 False → 贪心解码（greedy decoding），输出确定性高，适合知识问答
    # #                 True  → 随机采样，输出更多样但可能不稳定
    # # 返回值: (response, history)
    # #   response: 模型生成的文本回答
    # #   history:  更新后的对话历史（包含本轮问答）
    # response, _ = model.chat(tokenizer, prompt, history=[], max_length=512, do_sample=False)
    # return response
    model = ChatOpenAI(
        model=os.environ.get("LLM_MODEL"),
        base_url=os.environ.get("DASHSCOPE_BASE_URL"),
        openai_api_key=os.environ.get("DASHSCOPE_API_KEY"),
        max_tokens=1000,
        temperature=0.1
    )

    result = model.invoke(prompt)
    return result

# ============================================================================
# 主流程
# ============================================================================

def main():
    """
    主入口函数 —— 编排完整的 RAG 查询流程。

    执行流程：
      (1) 解析命令行参数
      (2) 加载 BGE-M3 嵌入模型
      (3) 连接 Milvus 并初始化数据
      (4) 混合召回 —— 标题向量 + 正文向量加权搜索
      (5) 重排序  —— 交叉编码器对候选精细打分
      (6) 构建 Prompt —— 将上下文和问题拼装为结构化指令
      (7) 大模型生成 —— ChatGLM-6B 基于资料回答（或仅打印 Prompt）
    """
    # ----- (1) 命令行参数解析 -----
    parser = argparse.ArgumentParser(
        description="RAG 混合召回 + Rerank 重排序 + 大模型问答 Demo"
    )
    parser.add_argument(
        "--query",
        default="入职需要提交哪些材料？",
        help="用户查询问题",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="重建 Milvus Collection（删除已有数据，重新创建并写入）",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="仅输出构造好的 Prompt，不调用大模型（方便调试召回和排序结果）",
    )
    args = parser.parse_args()

    print(args.query)
    print(args.rebuild)
    print(args.no_llm)
    # exit(0)
    # ----- (2) 加载 BGE-M3 嵌入模型 -----
    # 延迟导入：FlagEmbedding 可能安装失败或不需要在启动时就加载
    # use_fp16=False: 使用 FP32 精度（CPU 推理更稳定，无 GPU 时可用）
    from FlagEmbedding import BGEM3FlagModel

    embed_model = BGEM3FlagModel(str(EMBED_MODEL_PATH), use_fp16=False)

    # ----- (3) 连接 Milvus 并准备数据 -----
    client = connect_client()                                    # 连接 Milvus 服务
    prepare_milvus(client, embed_model, rebuild=args.rebuild)    # 初始化 Collection 并写入文档

    # ----- (4) 混合召回 -----
    # 同时搜索标题向量和正文向量，加权融合得到候选文档
    recalled_hits = hybrid_search(client, embed_model, args.query)
    print("\n=== Milvus 混合召回 ===")
    for hit in recalled_hits:
        # recall_score: 加权融合后的最终分数（余弦相似度加权和）
        print(f"{hit['recall_score']:.4f} | {hit['category']} | {hit['title']} | {hit['content']}")

    # ----- (5) 重排序 -----
    # 用交叉编码器对候选文档重新打分，选出最相关的 TopK
    reranked_hits = rerank(args.query, recalled_hits)
    print("\n=== Rerank 重排序 ===")
    for hit in reranked_hits:
        # rerank_score: 交叉编码器给出的深度语义相关性分数
        print(f"{hit['rerank_score']:.4f} | {hit['category']} | {hit['title']}")

    # ----- (6) 构建 Prompt -----
    prompt = build_prompt(args.query, reranked_hits)

    # ----- (7) 输出结果或调用大模型 -----
    # --no-llm: 调试模式 —— 只打印 Prompt，不调用模型
    if args.no_llm:
        print("\n=== Prompt ===")
        print(prompt)
        return  # 提前退出，不执行模型推理

    # 正常模式: 将 Prompt 送入 ChatGLM-6B，生成最终自然语言答案
    answer = ask_llm(prompt)
    print("\n=== 最终答案 ===")
    print(answer)


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    main()
