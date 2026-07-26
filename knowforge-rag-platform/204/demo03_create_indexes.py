"""
演示为 Collection 的向量字段创建 HNSW 索引的完整流程：
  1. 创建 Collection（如果没有）
  2. 为 dense 向量字段创建 HNSW 索引
  3. 查看索引详情

学习要点：
  1. 理解"创建 Collection ≠ 有索引"——必须显式调用 create_index()
  2. 掌握 HNSW 的三个关键参数：M、efConstruction、metric_type
  3. 能通过 list_indexes 和 describe_index 查看索引状态
"""
from milvus_common import BASE_COLLECTION, connect_client, create_hnsw_index, recreate_base_collection

def main():
    # ── 步骤 1：获取客户端并确保 Collection 存在 ──
    client = connect_client()
    # 重新创建 Collection，确保索引示例在干净的 Schema 上运行
    recreate_base_collection(client, collection_name=BASE_COLLECTION, drop_old=True)

    # ── 步骤 2：创建 HNSW 索引 ──
    # create_hnsw_index 内部做的事：
    #   1. 检查索引是否已存在（幂等）
    #   2. client.prepare_index_params() → 创建索引参数容器
    #   3. index_params.add_index(field_name="dense", ...) → 添加 HNSW 配置
    #   4. client.create_index(...) → 提交索引构建请求
    #
    # HNSW 参数详解：
    #   M = 16
    #     每个节点在图中最多连接 16 个邻居
    #     取值范围 [4, 64]，默认 16
    #     ↑ 增大 → 精度提高 + 内存增加（图更密）
    #     ↓ 减小 → 速度更快 + 内存减少（图更稀）
    #
    #   efConstruction = 100
    #     构建索引时的搜索宽度
    #     取值范围 [8, 512]，默认 200
    #     ↑ 增大 → 索引质量提高 + 构建时间变长
    #
    #   metric_type = "COSINE"
    #     向量相似度度量方式
    #     COSINE：余弦相似度 [-1, 1]，1 表示方向完全一致
    #     IP：内积，向量归一化后等价于余弦
    #     L2：欧几里得距离，值越小越相似
    #
    # ⚠️ create_index 是异步操作！
    #   调用后立即返回，Milvus 在后台构建索引
    #   构建期间数据仍可查询（走 Growing Segment 暴力搜索）
    create_hnsw_index(client, collection_name=BASE_COLLECTION)

    # ── 步骤 3：查看索引详情 ──
    # list_indexes：列出该 Collection 所有索引的名称
    # describe_index：查看指定索引的详细配置（类型、度量、参数）
    print("\n=== 当前索引列表 ===")
    for index_name in client.list_indexes(collection_name=BASE_COLLECTION):
        detail = client.describe_index(collection_name=BASE_COLLECTION, index_name=index_name)
        # detail 包含：index_name, field_name, index_type, metric_type, params
        print(detail)


if __name__ == "__main__":
    main()
