"""
演示 Milvus 中创建数据表（Collection）的完整流程：
  1. Schema 设计：定义字段（主键、向量、标量）
  2. Collection(集合) 创建：将 Schema 绑定到物理存储
  3. describe_collection：查看 Collection 的完整元数据

学习要点：
  1. 理解 Schema = "表结构定义"，Collection = "物理表"
  2. 掌握 auto_id、enable_dynamic_field、consistency_level 的含义
  3. 能区分 VARCHAR / FLOAT_VECTOR / INT64 等数据类型的适用场景
"""

from __future__ import annotations

from milvus_common import BASE_COLLECTION, connect_client, recreate_base_collection


def main():
    # ── 步骤 1：获取 MilvusClient ──
    client = connect_client()

    # ── 步骤 2：创建 Collection（使用 milvus_common 中预定义的 Schema）──
    # drop_old=True：如果 Collection 已存在则先删除再重建
    #   ⚠️ 仅用于教学 demo，生产环境不会轻易 drop 已有数据
    #
    # recreate_base_collection 内部做的事（详见 milvus_common.py）：
    #   1. 如果 drop_old=True → client.drop_collection()
    #   2. 如果 Collection 已存在 → 直接返回（幂等）
    #   3. client.create_schema() → 定义字段
    #   4. client.create_collection(schema, consistency_level="Session")
    #
    # 本 demo 的 Schema 包含 5 个字段：
    #   pk         VARCHAR(128)  主键（is_primary=True, auto_id=False）
    #   text       VARCHAR(2048) 文本内容
    #   dense      FLOAT_VECTOR(dim=5) 稠密向量
    #   source     VARCHAR(64)   业务分类（标量过滤）
    #   kb_version VARCHAR(64)   知识库版本（版本过滤）
    #
    # auto_id=False 的含义：
    #   主键由用户手动指定，不使用 Milvus 自增 ID
    #   项目中使用 chunk_id / faq_id 作为主键，支持精确的更新和删除操作
    #
    # enable_dynamic_field=True 的含义：
    #   插入数据时可以携带 Schema 中未定义的额外字段
    #   这些字段存入 $meta 特殊字段，仍可被过滤表达式查询
    recreate_base_collection(client, collection_name=BASE_COLLECTION, drop_old=True)

    # ── 步骤 3：查看 Collection 完整信息 ──
    # describe_collection 返回的字典包含：
    #   - collection_name：Collection 名称
    #   - schema：字段定义列表
    #   - auto_id：是否自动生成主键
    #   - enable_dynamic_field：是否允许动态字段
    #   - consistency_level：一致性级别
    #   - indexes：已创建的索引列表
    description = client.describe_collection(collection_name=BASE_COLLECTION)
    print("\n=== Collection 描述 ===")
    print(description)


if __name__ == "__main__":
    main()
