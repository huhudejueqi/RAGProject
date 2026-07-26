"""
演示基于pymilvus Milvus 的多向量字段混合搜索：
  1. 一个 Collection 包含两个 Dense 向量字段（filmVector + posterVector）
  2. 每个字段独立建 HNSW 索引
  3. AnnSearchRequest 封装单路搜索请求
  4. WeightedRanker 加权融合两路结果

这是第8讲 Dense+Sparse 混合检索的教学简化版：
  - 本 demo：两个 Dense 向量（模拟多特征搜索）
  - 一个 Dense（BGE-M3 语义向量）+ 一个 Sparse（BM25 关键词向量）

学习要点：
  1. 掌握 AnnSearchRequest 的封装方式
  2. 理解 WeightedRanker 的权重融合机制
  3. 能区分 search() 和 hybrid_search() 的适用场景
"""
from pymilvus import AnnSearchRequest, DataType, WeightedRanker
from milvus_common import MILVUS_DB, connect_client, normalize

COLLECTION_NAME = "lecture04_hybrid_demo"


def create_collection(client):
    """创建包含两个向量字段的 Collection。

    Schema 设计：
      id           INT64 主键
      title        VARCHAR 文档标题
      filmVector   FLOAT_VECTOR(5) 模拟语义特征向量
      posterVector FLOAT_VECTOR(5) 模拟另一个特征空间的向量
      category     VARCHAR 业务分类
    """
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=256)
    # 两个向量字段，模拟不同的特征维度（如语义 vs 视觉）
    schema.add_field(field_name="filmVector", datatype=DataType.FLOAT_VECTOR, dim=5)
    schema.add_field(field_name="posterVector", datatype=DataType.FLOAT_VECTOR, dim=5)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=64)
    client.create_collection(
        collection_name=COLLECTION_NAME, schema=schema, consistency_level="Session",
    )

    # 为两个向量字段各建一个 HNSW 索引
    index_params = client.prepare_index_params()
    for field_name in ["filmVector", "posterVector"]:
        index_params.add_index(
            field_name=field_name,
            index_name=f"{field_name}_hnsw",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 100},
        )
    client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)

def insert_data(client):
    """插入 3 条示例数据并加载到内存。

    设计意图：
      - id=1（risk）→ 第三维较大（0.92），表示贸易/风险类文档
      - id=2（hr）  → 第一维较大（0.92），表示 HR 类文档
      - id=3（fin）  → 第二维较大（0.92），表示财务类文档
    同一文档的 filmVector 和 posterVector 各维度值接近（模拟同一文档的多视角特征）
    """
    client.insert(
        collection_name=COLLECTION_NAME,
        data=[
            {
                "id": 1, "title": "合规风险案例库",
                "filmVector": normalize([0.1, 0.1, 0.92, 0.2, 0.05]),
                "posterVector": normalize([0.08, 0.12, 0.88, 0.24, 0.06]),
                "category": "risk",
            },
            {
                "id": 2, "title": "员工入职手册",
                "filmVector": normalize([0.92, 0.12, 0.08, 0.03, 0.02]),
                "posterVector": normalize([0.86, 0.16, 0.1, 0.05, 0.03]),
                "category": "hr",
            },
            {
                "id": 3, "title": "费用报销指南",
                "filmVector": normalize([0.08, 0.92, 0.12, 0.05, 0.02]),
                "posterVector": normalize([0.12, 0.88, 0.14, 0.06, 0.02]),
                "category": "finance",
            },
        ],
    )
    # 加载 Collection 到内存（必须先 load 才能 search）
    client.load_collection(collection_name=COLLECTION_NAME)

def main():
    # 1：创建客户端实例
    client = connect_client()
    # 2：创建表以及索引
    create_collection(client)
    # 3: 插入数据
    insert_data(client)

    """
    pymilvus的混合检索实现
    双路召回（稠密+稀疏）AnnSearchRequest(封装了单路向量搜索的全部参数)
    
    参数说明：
        data：查询向量（自动归一化）
        filmVector: 在哪个向量字段上搜索
        param：搜索参数
        limit: 这一路返回多少候选
    """

    # 请求1：在filmVector向量字段上搜索
    film_request = AnnSearchRequest(
        data=[normalize([0.08,0.1,0.94,0.22,0.04])],
        anns_field="filmVector",
        param={"metric_type":"COSINE", "params":{"ef":64}},
        limit=3
    )

    # 请求2：在posterVector向量字段上搜索
    poster_request = AnnSearchRequest(
        data=[normalize([0.1, 0.1, 0.88, 0.28, 0.05])],
        anns_field="posterVector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=3
    )

    """"
    WeightedRanker：加权融合
    WeightedRanker(w1, w2)：将两路候选按照权重加权融合派和
    - filmVector 权重0.55（语义特征主导）
    - posterVector 权重0.45（关键词辅助特征）
    """
    result = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[film_request, poster_request],
        ranker=WeightedRanker(0.55, 0.45),
        limit=3,
        output_fields=["title","category"]
    )

    print("多向量字段混合搜索----------")
    print(result)

if __name__ == '__main__':
    main()
