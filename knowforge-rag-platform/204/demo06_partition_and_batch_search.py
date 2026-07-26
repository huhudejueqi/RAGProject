"""
演示两个进阶功能：
  1. Partition（分区）搜索：只搜索指定分区内的数据，减少扫描量
  2. 批量向量搜索：一次请求传入多个查询向量

学习要点：
  1. 理解 Partition 的分区裁剪原理
  2. 掌握批量搜索的输入格式（N个向量）和输出格式（N×limit 条结果）
"""

from __future__ import annotations

from milvus_common import (
    BASE_COLLECTION, connect_client, create_hnsw_index,
    load_base_collection, normalize, recreate_base_collection,
)

# 分区名称：将 trade v2 版本的数据单独隔离
PARTITION_NAME = "trade_v2"

def main():
    client = connect_client()
    # 从头开始：先删除旧 Collection，再重建
    recreate_base_collection(client, collection_name=BASE_COLLECTION, drop_old=True)
    create_hnsw_index(client, collection_name=BASE_COLLECTION)

    # ========================================================================
    # 一、创建分区并插入分区数据
    # ========================================================================
    # Partition 将 Collection 内数据按业务维度物理隔离
    # 搜索时指定 partition_names → 只扫描该分区 → 减少搜索计算量
    if PARTITION_NAME not in client.list_partitions(collection_name=BASE_COLLECTION):
        client.create_partition(collection_name=BASE_COLLECTION, partition_name=PARTITION_NAME)

    # 向指定分区插入数据（共 2 条贸易类文档）
    client.insert(
        collection_name=BASE_COLLECTION,
        partition_name=PARTITION_NAME,           # 指定分区名
        data=[
            {
                "pk": "trade_p_001",
                "text": "跨境贸易合同需要检查交易对手、商品编码和贸易管制要求。",
                "dense": normalize([0.05, 0.08, 0.95, 0.18, 0.04]),
                "source": "trade",
                "kb_version": "v2",
            },
            {
                "pk": "trade_p_002",
                "text": "命中制裁名单时应停止出货并转人工合规审批。",
                "dense": normalize([0.02, 0.05, 0.9, 0.28, 0.06]),
                "source": "trade",
                "kb_version": "v2",
            },
        ],
    )
    # 加载到内存（包含分区数据）
    load_base_collection(client, collection_name=BASE_COLLECTION)

    # ========================================================================
    # 二、分区 + 批量搜索
    # ========================================================================
    # 这次搜索做了两件事：
    #   1. partition_names=["trade_v2"] → 只扫描指定分区
    #   2. data=[query1, query2] → 批量传入 2 个查询向量
    #
    # 批量搜索的返回值结构：
    #   results[0] → 第一个查询向量的所有命中（按分数降序）
    #   results[1] → 第二个查询向量的所有命中
    results = client.search(
        collection_name=BASE_COLLECTION,
        partition_names=[PARTITION_NAME],        # 只搜索此分区
        data=[
            normalize([0.05, 0.08, 0.96, 0.16, 0.04]),  # 查询 1
            normalize([0.02, 0.04, 0.86, 0.32, 0.05]),  # 查询 2
        ],
        anns_field="dense",
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=2,                                 # 每个查询返回 Top-2
        output_fields=["pk", "text"],
    )

    print("=== 分区 + 批量搜索 ===")
    for query_index, hits in enumerate(results, start=1):
        print(f"\nQuery {query_index}")
        for hit in hits:
            print(hit)


if __name__ == "__main__":
    main()
