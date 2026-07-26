"""
milvus_common.py — Milvus 教学演示的共享工具模块
==================================================
对应讲次：第4讲《Milvus 索引机制与基本操作》

本模块为 demo01~demo12 提供统一的：
  1. Milvus 连接配置（URI / Database / Collection 名称）
  2. 向量归一化工具函数
  3. 5 条模拟企业文档（HR / 财务 / 贸易三个场景，含 pk、text、dense 向量、source、kb_version）
  4. MilvusClient 连接工厂（自动创建 Database）
  5. Schema 工厂、Collection 重建、HNSW 索引创建、加载、示例数据 Upsert

学生无需在每个 demo 中重复编写连接和建表代码，
导入本模块即可快速进入"索引/检索/过滤/混合搜索"等核心操作。
"""

from __future__ import annotations

import math
import os
from typing import Iterable, List

# ---------------------------------------------------------------------------
# pymilvus 是 Milvus 官方 Python SDK
#   - MilvusClient：新版高层 API（v2.4+ 推荐），连接、建表、建索引、搜索统一入口
#   - DataType：枚举类型，用于定义字段的数据类型（VARCHAR / INT64 / FLOAT_VECTOR 等）
# ---------------------------------------------------------------------------
from pymilvus import DataType, MilvusClient

# ============================================================================
# 一、连接配置（优先从环境变量读取，方便 CI / 多机部署）
# ============================================================================
MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
"""Milvus 服务地址。默认连接本地 Standalone 实例的 gRPC 端口 19530。"""

MILVUS_DB = os.getenv("MILVUS_DATABASE", "milvus_demo")
"""Milvus 逻辑数据库名称。Milvus 2.4+ 支持多 Database（类似 MySQL 的 database 概念），
本课程所有 demo 共用同一个 Database，避免污染用户的默认数据库。"""

BASE_COLLECTION = os.getenv("MILVUS_COLLECTION", "wh05_lecture04_pymilvus_demo")
"""默认 Collection 名称。Collection 是 Milvus 中存储数据的"表"，
同一 Collection 内的数据共享相同的 Schema 和索引配置。"""

DIM = 5
"""向量维度。教学演示使用 5 维（而非真实 BGE-M3 的 1024 维），
以便直观观察向量数值和距离计算过程。"""


# ============================================================================
# 二、向量归一化工具
# ============================================================================
def normalize(values: Iterable[float]) -> List[float]:
    """将向量 L2 归一化到单位长度。

    归一化后的向量 ||v|| = 1，此时：
      - 内积（IP）在数学上等价于余弦相似度（Cosine）
      - Milvus 默认使用内积作为度量，归一化后效果等同于余弦

    这在项目中对应 BGE-M3 的 normalize_embeddings=True 设置。

    在真实的业务中不需要手动对向量进行标准化，而是使用Embedding实现

    参数：
        values: 任意可迭代的浮点数序列（原始向量）

    返回：
        归一化后的浮点数列表，保留 6 位小数。

    示例：
        >>> normalize([3.0, 4.0])
        [0.6, 0.8]  # 3²+4²=25, sqrt=5, 3/5=0.6, 4/5=0.8
    """
    values = list(values)
    # 计算 L2 范数：sqrt(Σ(vᵢ²))
    norm = math.sqrt(sum(item * item for item in values)) or 1.0
    # 每个维度除以范数，使向量长度 = 1
    return [round(item / norm, 6) for item in values]


# ============================================================================
# 三、模拟企业文档数据（SAMPLE_DOCS）
# ============================================================================
# 共 5 条数据，覆盖 3 个业务场景：
#   - HR（入职/离职）2 条
#   - 财务（报销）1 条
#   - 贸易（跨境申报/制裁）2 条
#
# 每条数据包含：
#   - pk：主键，唯一标识一条文档
#   - text：文本内容（会被 BGE-M3 转为 Dense 向量，被 Milvus BM25 转为 Sparse 向量）
#   - dense：预归一化的 Dense 向量（5 维，教学用；真实场景是 1024 维）
#   - source：业务分类标签（用于标量过滤，如 source == "hr"）
#   - kb_version：知识库版本号（用于版本过滤和增量更新）
#
# 设计意图：
#   - 同一 source 的向量彼此接近（语义相近），不同 source 的向量远离
#   - 例如 hr 的向量第一维很大、finance 的第二维很大、trade 的第三维很大
#   - 这模拟了真实场景中"同类文档向量聚集"的现象
SAMPLE_DOCS = [
    {
        "pk": "doc_hr_001",
        "text": "员工入职需要提交身份证、学历证明、银行卡和体检报告。",
        "dense": normalize([0.95, 0.12, 0.08, 0.03, 0.02]),
        "source": "hr",
        "kb_version": "v1",
    },
    {
        "pk": "doc_hr_002",
        "text": "离职交接需要完成资产归还、账号注销和工作交接确认。",
        "dense": normalize([0.86, 0.2, 0.12, 0.08, 0.04]),
        "source": "hr",
        "kb_version": "v1",
    },
    {
        "pk": "doc_fin_001",
        "text": "费用报销需要提供发票、审批单和付款凭证。",
        "dense": normalize([0.08, 0.92, 0.14, 0.06, 0.03]),
        "source": "finance",
        "kb_version": "v1",
    },
    {
        "pk": "doc_trade_001",
        "text": "跨境申报要检查 HS 编码、原产地证明和许可证要求。",
        "dense": normalize([0.08, 0.1, 0.93, 0.12, 0.04]),
        "source": "trade",
        "kb_version": "v2",
    },
    {
        "pk": "doc_trade_002",
        "text": "制裁名单命中后应暂停交易并提交合规复核。",
        "dense": normalize([0.05, 0.1, 0.88, 0.24, 0.07]),
        "source": "trade",
        "kb_version": "v2",
    },
]


# ============================================================================
# 四、MilvusClient 连接工厂
# ============================================================================
def connect_client() -> MilvusClient:
    """创建 MilvusClient 并切换到演示 Database。

    MilvusClient 是 pymilvus 2.4+ 推荐的高层 API：
      - 内部管理 gRPC 连接池
      - 提供 create_collection / insert / search / delete 等一站式方法
      - 线程安全，可在多线程环境中复用

    执行流程：
      1. 创建 MilvusClient 实例（连接 Milvus 服务端）
      2. 如果 MILVUS_DB 数据库不存在，自动创建
      3. 切换到 MILVUS_DB，后续所有操作在此 Database 下进行

    返回：
        已连接并切换到演示 Database 的 MilvusClient 实例。
    """
    # 创建客户端（默认连接 localhost:19530）
    # 如果milvus安装在本地的docker环境那么地址就是localhost:19530，
    # 如果milvus安装在虚拟机的docker环境那么地址就是192.168.88.100:19530
    client = MilvusClient(uri=MILVUS_URI)
    if MILVUS_DB:
        # 检查 Database 是否存在
        if MILVUS_DB not in client.list_databases():
            # 不存在则创建（Milvus Database 是逻辑隔离单元，类似 MySQL 的 CREATE DATABASE）
            client.create_database(MILVUS_DB)
        # 切换到演示 Database（类似 USE database）。
        # PyMilvus 2.6+ 将 using_database 标记为弃用，优先使用新 API。
        if hasattr(client, "use_database"):
            client.use_database(MILVUS_DB)
        else:
            client.using_database(MILVUS_DB)
    return client

# ============================================================================
# 五、Schema 工厂
# ============================================================================
def create_base_schema(client: MilvusClient):
    """创建基础 Schema（不含索引定义，仅定义字段结构）。

    Schema 是 Collection 的"表结构"，定义了：
      - 有哪些字段（Field）
      - 每个字段的类型（DataType.VARCHAR / FLOAT_VECTOR 等）
      - 哪个字段是主键
      - 是否允许动态字段

    本 Schema 包含 5 个字段：
      pk        → VARCHAR(128) 主键，手动指定（auto_id=False）
      text      → VARCHAR(2048) 文本内容
      dense     → FLOAT_VECTOR(5) 稠密向量（真实场景为 1024 维）
      source    → VARCHAR(64) 业务分类（用于标量过滤 WHERE source == "hr"）
      kb_version → VARCHAR(64) 知识库版本（用于版本过滤）

    参数：
        client: MilvusClient 实例

    返回：
        pymilvus 的 Schema 对象（尚未关联到任何 Collection）。
    """
    # auto_id=False：主键由用户手动指定（本项目使用 chunk_id / faq_id 作为主键）
    # enable_dynamic_field=True：允许插入时携带 Schema 中未定义的额外字段
    #   （这些字段存入 $meta 字段，可被过滤表达式查询）
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    # ── 添加字段 ──
    # is_primary=True：标记为主键，值必须唯一
    schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
    # 文本字段：存储原始文档内容，被 BGE-M3 Embedding 和 Milvus BM25 共同使用
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2048)
    # 稠密向量字段：BGE-M3 生成的 1024 维浮点向量（教学用 5 维）
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=DIM)
    # 标量字段：用于过滤（如只检索 HR 部门的文档）
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="kb_version", datatype=DataType.VARCHAR, max_length=64)
    return schema


def _describe_collection_fields(client: MilvusClient, collection_name: str) -> dict[str, dict]:
    """把 describe_collection 返回的 fields 列表整理成按字段名索引的字典。"""
    detail = client.describe_collection(collection_name=collection_name)
    return {field["name"]: field for field in detail.get("fields", [])}


def _collection_row_count(client: MilvusClient, collection_name: str) -> int:
    """读取 Collection 当前已有的数据行数。"""
    stats = client.get_collection_stats(collection_name=collection_name)
    return int(stats.get("row_count", 0))


def _base_collection_schema_mismatch(client: MilvusClient, collection_name: str) -> list[str]:
    """检查当前 Collection 是否和本项目的基础 Schema 兼容。"""
    fields = _describe_collection_fields(client, collection_name)
    mismatches: list[str] = []

    required_fields = {
        "pk": DataType.VARCHAR,
        "text": DataType.VARCHAR,
        "dense": DataType.FLOAT_VECTOR,
        "source": DataType.VARCHAR,
        "kb_version": DataType.VARCHAR,
    }

    for field_name, expected_type in required_fields.items():
        field = fields.get(field_name)
        if field is None:
            mismatches.append(f"缺少字段 {field_name!r}")
            continue
        if field.get("type") != expected_type:
            mismatches.append(
                f"字段 {field_name!r} 类型不一致：当前 {field.get('type')}，期望 {expected_type}"
            )

    pk_field = fields.get("pk")
    if pk_field and not pk_field.get("is_primary", False):
        mismatches.append("字段 'pk' 不是主键")

    dense_field = fields.get("dense")
    if dense_field:
        current_dim = dense_field.get("params", {}).get("dim")
        if str(current_dim) != str(DIM):
            mismatches.append(f"字段 'dense' 维度不一致：当前 {current_dim}，期望 {DIM}")

    return mismatches


# ============================================================================
# 六、Collection 重建工具
# ============================================================================
def recreate_base_collection(
    client: MilvusClient,
    collection_name: str = BASE_COLLECTION,
    drop_old: bool = False,
) -> None:
    """创建或重建演示 Collection。

    设计为幂等操作：如果 Collection 已存在且未指定 drop_old=True，
    则直接复用（避免每次运行 demo 都重建）。

    参数：
        client: MilvusClient 实例
        collection_name: 目标 Collection 名称
        drop_old: 是否删除已有 Collection（⚠️ 数据不可恢复！本项目默认 False）

    执行流程：
      1. 如果 drop_old=True 且 Collection 存在 → 删除
      2. 如果 Collection 已存在 → 直接返回（幂等）
      3. 创建 Schema → create_collection（指定一致性级别为 Session）
    """
    # 强制重建模式：先删后建（⚠️ 数据丢失）
    if drop_old and client.has_collection(collection_name):
        client.drop_collection(collection_name)

    if client.has_collection(collection_name):
        mismatches = _base_collection_schema_mismatch(client, collection_name)
        if not mismatches:
            print(f"Collection already exists: {collection_name}")
            return

        row_count = _collection_row_count(client, collection_name)
        mismatch_text = "; ".join(mismatches)
        if row_count == 0:
            print(
                f"Collection schema is outdated but empty: {collection_name}。"
                f"发现问题：{mismatch_text}。正在自动重建。"
            )
            client.drop_collection(collection_name)
        else:
            raise RuntimeError(
                f"Collection {collection_name!r} 的 Schema 与当前代码不兼容，且已有 {row_count} 条数据。"
                f"发现问题：{mismatch_text}。"
                "请先备份数据，再手动 drop/recreate，或指定 drop_old=True。"
            )

    # 创建 Schema
    schema = create_base_schema(client)
    # 创建 Collection
    # consistency_level="Session"：会话一致性，保证本客户端写入后立即可查
    #   （比 Strong 快，比 Eventually 可靠，适合单机教学场景）
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        consistency_level="Session",
    )
    print(f"Created collection: {collection_name}")


# ============================================================================
# 七、HNSW 索引创建
# ============================================================================
def create_hnsw_index(client: MilvusClient, collection_name: str = BASE_COLLECTION) -> None:
    """为 Collection 的 dense 向量字段创建 HNSW 索引。

    索引是 Milvus 实现快速 ANN（近似最近邻）搜索的核心：
      - 没有索引 → FLAT 暴力搜索（O(N×D)，精确但慢）
      - 有索引 → HNSW 图搜索（O(log N)，近似但快 10-1000 倍）

    HNSW 参数说明：
      M=16             每个节点最多连接 16 个邻居（越大→精度↑内存↑）
      efConstruction=100 构建时的搜索宽度（越大→索引质量↑构建时间↑）
      metric_type="COSINE" 余弦相似度度量

    参数：
        client: MilvusClient 实例
        collection_name: 目标 Collection 名称

    执行流程：
      1. 检查索引是否已存在（幂等）
      2. 创建 IndexParams → 添加 HNSW 索引配置 → 调用 create_index
    """
    field_name = "dense"
    index_name = "dense_hnsw"
    index_type = "HNSW"
    metric_type = "COSINE"
    index_params_values = {"M": 16, "efConstruction": 100}

    # 幂等判断必须按字段做，而不是只按索引名做。
    # Milvus 不允许同一个字段同时存在多个索引；如果 dense 字段已有名为 dense 的索引，
    # 再创建 dense_hnsw 也会失败。
    index_names = client.list_indexes(collection_name=collection_name)
    # 获取到表中的行的数据数量
    row_count = _collection_row_count(client, collection_name)
    for existing_index_name in index_names:
        # 根据表（集合）名和索引的名字获取到索引的详细信息
        detail = client.describe_index(
            collection_name=collection_name,
            index_name=existing_index_name,
        )
        # 索引关联的列的名字，如果这个列添加过索引，则跳过
        if detail.get("field_name") != field_name:
            continue

        same_index = (
            detail.get("index_type") == index_type
            and detail.get("metric_type") == metric_type
            and str(detail.get("M")) == str(index_params_values["M"])
            and str(detail.get("efConstruction")) == str(index_params_values["efConstruction"])
        )
        if same_index:
            print(f"Index already exists on field '{field_name}': {existing_index_name}")
            return

        if row_count == 0:
            print(
                f"Collection is empty, dropping incompatible index '{existing_index_name}' "
                f"on field '{field_name}' and recreating."
            )
            client.drop_index(collection_name=collection_name, index_name=existing_index_name)
            break

        raise RuntimeError(
            f"字段 '{field_name}' 已存在索引 '{existing_index_name}'，"
            f"配置为 index_type={detail.get('index_type')}, "
            f"metric_type={detail.get('metric_type')}, "
            f"M={detail.get('M')}, efConstruction={detail.get('efConstruction')}。"
            "Milvus 不支持同一个字段同时创建多个索引；如需更换索引配置，"
            "请先删除旧索引或重建 Collection。"
        )

    # 准备索引参数容器
    index_params = client.prepare_index_params()
    # 为 dense 字段添加 HNSW 索引
    index_params.add_index(
        field_name=field_name,        # 对哪个向量字段建索引
        index_name=index_name,        # 索引名称（用于后续管理）
        index_type=index_type,        # 索引类型：分层可导航小世界图
        metric_type=metric_type,      # 相似度度量：余弦相似度
        params=index_params_values,   # HNSW 构建参数
    )
    # 调用 create_index，Milvus 后台异步构建索引
    client.create_index(collection_name=collection_name, index_params=index_params)
    print(f"Created vector index: {index_name}")


# ============================================================================
# 八、Collection 加载到内存
# ============================================================================
def load_base_collection(client: MilvusClient, collection_name: str = BASE_COLLECTION) -> None:
    """将 Collection 的索引和数据加载到内存。

    ⚠️ 关键：必须先 load 才能执行 search/query！
      - 未 load → 数据在磁盘上 → search 会报错
      - load 后 → 索引 + 数据在内存中 → 毫秒级搜索

    load_collection 内部：
      1. 把索引文件从 MinIO 加载到内存
      2. 把数据 segment 加载到内存
      3. 此后搜索走内存中的索引结构（HNSW 图）

    参数：
        client: MilvusClient 实例
        collection_name: 目标 Collection 名称
    """
    client.load_collection(collection_name=collection_name)
    print(f"Loaded collection: {collection_name}")


# ============================================================================
# 九、示例数据 Upsert
# ============================================================================
def upsert_sample_docs(client: MilvusClient, collection_name: str = BASE_COLLECTION) -> None:
    """将 SAMPLE_DOCS 写入 Collection（幂等 Upsert）。

    Upsert = Update + Insert：
      - 主键已存在 → 更新该条数据
      - 主键不存在 → 插入新数据
    这保证多次运行 demo 不会产生重复数据。

    执行流程：
      1. client.upsert() → 数据写入 Milvus
      2. client.flush() → 强制持久化到磁盘（避免数据留在内存缓冲区）

    参数：
        client: MilvusClient 实例
        collection_name: 目标 Collection 名称
    """
    # 写入数据（Dense 向量已预归一化，Sparse 向量由 Milvus BM25 在服务端自动生成）
    client.upsert(collection_name=collection_name, data=SAMPLE_DOCS)
    # flush 将内存中的 segment 持久化到对象存储（MinIO）
    # 兼容 pymilvus 不同版本的 flush API 差异
    try:
        client.flush(collection_name=collection_name)
    except TypeError:
        # 某些版本 flush 接受位置参数而非关键字参数
        client.flush(collection_name)
    print(f"Upserted {len(SAMPLE_DOCS)} sample documents")
