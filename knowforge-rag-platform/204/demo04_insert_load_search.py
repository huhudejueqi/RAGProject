"""
演示 Milvus 核心操作三连：插入（Insert）→ 加载（Load）→ 搜索（Search）
  1. Upsert 示例数据（5 条模拟企业文档）
  2. load_collection 将索引和数据加载到内存
  3. 执行 ANN 向量搜索 + 标量过滤 + 输出指定字段

学习要点：
  1. 理解 insert → load → search 的依赖顺序（必须先 load 才能 search）
  2. 掌握 search() 的参数含义：anns_field / metric_type / ef / filter / output_fields
  3. 能将 Milvus 搜索结果解析为 (score, entity) 格式
"""

from milvus_common import (
    BASE_COLLECTION,
    connect_client,
    create_hnsw_index,
    load_base_collection,
    normalize,
    recreate_base_collection,
    upsert_sample_docs,
)

def main():
    # ── 步骤 1：准备环境（连接 → 建表 → 建索引 → 写数据 → 加载）──
    client = connect_client()
    # 重新创建 Collection，避免沿用旧 Schema 或旧数据
    recreate_base_collection(client, collection_name=BASE_COLLECTION, drop_old=True)
    # 为 dense 向量字段创建 HNSW 索引
    create_hnsw_index(client, collection_name=BASE_COLLECTION)

    # 写入 5 条模拟企业文档（Upsert = 幂等写入）
    upsert_sample_docs(client, collection_name=BASE_COLLECTION)

    # ⚠️ 关键步骤：将 Collection 加载到内存
    #   未 load 的 Collection 无法 search！load 后索引和数据进入内存，开始加速
    load_base_collection(client, collection_name=BASE_COLLECTION)

    # ── 步骤 2：构造查询向量 ──
    # 模拟"查询贸易相关文档"的场景
    # 该向量的第 3 维（索引 2）较大（0.95），与 SAMPLE_DOCS 中 trade 文档的第 3 维（0.93/0.88）接近
    # normalize() 确保向量 L2 归一化，使 COSINE 距离计算准确
    query = "入职需要哪些流程？"
    query_vector = normalize([0.1, 0.1, 0.95, 0.12, 0.02])

    # ── 步骤 3：执行向量搜索 ──
    # MilvusClient.search() 参数详解：
    #
    #   collection_name  目标 Collection
    #   data            查询向量列表（支持批量搜索，传入多个向量）
    #   anns_field       在哪个向量字段上做 ANN 搜索（这里选 "dense" 稠密向量字段）
    #   search_params    搜索参数：
    #     metric_type    相似度度量方式（COSINE=余弦）
    #     params.ef      HNSW 查询时的搜索宽度（越大→精度越高→速度越慢）
    #   limit            返回 Top-K 个最相似的结果
    #   filter           标量过滤表达式（Milvus Boolean Expr）
    #                    "source == 'trade'" → 只搜索贸易类的文档
    #   output_fields    搜索结果中返回哪些字段的值
    results = client.search(
        collection_name=BASE_COLLECTION,
        data=[query_vector],
        anns_field="dense",                                # 在稠密向量字段上搜索
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},  # HNSW 搜索参数
        limit=3,                                           # 返回 Top-3
        filter='source == "trade"',                        # 只查贸易类文档
        output_fields=["text", "source", "kb_version"],    # 返回这些字段的值
    )

    # ── 步骤 4：解析搜索结果 ──
    # results 是一个二维列表：results[查询索引][排名]
    #   results[0] → 第一个查询向量的所有命中
    #   results[0][0] → 最相似的结果
    #
    # 每条命中（hit）包含：
    #   hit["distance"]  → 相似度分数（COSINE: 越接近 1 越相似）
    #   hit["id"]        → 主键值
    #   hit["entity"]    → 通过 output_fields 指定的字段值字典
    print("\n=== 向量检索结果 ===")
    for hit in results[0]:
        entity = hit.get("entity", {})
        print(
            f"score={hit['distance']:.4f} "
            f"source={entity.get('source')} "
            f"text={entity.get('text')}"
        )


if __name__ == "__main__":
    main()
