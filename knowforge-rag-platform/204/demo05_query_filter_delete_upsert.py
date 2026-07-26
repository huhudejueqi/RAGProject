"""
演示三种核心数据管理操作：
  1. query()   — 标量过滤查询（只按标量条件筛选，不涉及向量相似度）
  2. upsert()  — 幂等更新/插入（主键匹配则更新，否则插入）
  3. delete()  — 按主键或布尔表达式删除数据

学习要点：
  1. 区分 query（标量过滤）和 search（向量相似度搜索）
  2. 理解 Upsert = Update + Insert 的幂等语义
  3. 掌握 delete 的两种方式：按 ids 和按 filter 表达式
"""

from __future__ import annotations

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
    # ── 准备环境 ──
    client = connect_client()
    recreate_base_collection(client, collection_name=BASE_COLLECTION, drop_old=True)
    create_hnsw_index(client, collection_name=BASE_COLLECTION)
    upsert_sample_docs(client, collection_name=BASE_COLLECTION)
    load_base_collection(client, collection_name=BASE_COLLECTION)

    # ========================================================================
    # 操作 1：标量过滤查询（query）
    # ========================================================================
    # query() 是纯粹的标量过滤，不涉及向量相似度计算
    # 类似于 SQL 的：SELECT pk, text, source FROM collection WHERE source = "hr"
    #
    # 使用场景：
    #   - 查看某个 source 下有哪些文档
    #   - 按 kb_version 筛选特定版本的数据
    #   - 调试时验证数据是否正确写入
    #
    # filter 表达式语法（Milvus Boolean Expr）：
    #   source == "hr"           → 字符串等值比较
    #   kb_version != "v1"       → 不等于
    #   source in ["hr","finance"] → 多值匹配
    #   source like "h%"         → 前缀匹配
    print("=== 标量过滤查询 ===")
    rows = client.query(
        collection_name=BASE_COLLECTION,
        filter='source == "hr"',                     # 只查 HR 部门的文档
        output_fields=["pk", "text", "source"],      # 返回这些字段
    )
    for row in rows:
        print(row)

    # ========================================================================
    # 操作 2：Upsert — 幂等更新
    # ========================================================================
    # Upsert = Update + Insert：
    #   - 主键（pk）已存在 → 更新整条记录（覆盖所有字段）
    #   - 主键不存在 → 插入新记录
    #
    # 本示例：更新 doc_hr_002（离职文档），补充"权限回收"步骤
    #   - pk 保持 "doc_hr_002" 不变（识别为更新）
    #   - text 从原来的"离职交接需要完成..."改为"离职交接必须完成..."
    #   - kb_version 从 "v1" 更新为 "v2"
    #
    # ⚠️ 注意：Upsert 是整行替换，不是部分字段更新
    #   必须传入完整的字段值，未传入的字段会被设为默认值或清空
    print("\n=== Upsert：更新一条文档 ===")
    client.upsert(
        collection_name=BASE_COLLECTION,
        data=[
            {
                "pk": "doc_hr_002",
                "text": "离职交接必须完成资产归还、账号注销、权限回收和工作交接确认。",
                "dense": normalize([0.88, 0.18, 0.1, 0.08, 0.05]),
                "source": "hr",
                "kb_version": "v2",                  # 版本号更新
            }
        ],
    )
    # 验证更新结果
    print(client.query(
        collection_name=BASE_COLLECTION,
        filter='pk == "doc_hr_002"',
        output_fields=["*"],                         # * 表示返回所有字段
    ))

    # ========================================================================
    # 操作 3：Delete — 删除数据
    # ========================================================================
    # delete() 支持两种删除方式：
    #   方式一：ids=["pk1", "pk2"] — 按主键列表删除
    #   方式二：filter='source == "finance"' — 按布尔表达式批量删除
    #
    # ⚠️ 注意：
    #   - 删除是软删除（标记删除），数据不会立即从磁盘移除
    #   - Milvus 后台 Compaction 任务会异步物理清理已标记的数据
    #   - 删除后的数据在 Compaction 完成前仍占用存储空间
    print("\n=== Delete：删除一条文档 ===")
    # 按主键删除 doc_fin_001（财务文档）
    print(client.delete(collection_name=BASE_COLLECTION, ids=["doc_fin_001"]))
    # 验证删除：查询 source == "finance" 应返回空
    print(client.query(
        collection_name=BASE_COLLECTION,
        filter='source == "finance"',
        output_fields=["pk", "text"],
    ))

if __name__ == "__main__":
    main()
