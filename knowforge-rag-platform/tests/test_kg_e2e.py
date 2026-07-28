"""
端到端测试：知识图谱构建管线。

流程：
  1. 创建测试文档（模拟入职流程文档）
  2. 调用 run_knowledge_graph_pipeline 抽取实体+关系
  3. 查看抽取结果
  4. 验证 Milvus 中是否有数据写入
"""
import sys
import os

# 覆盖 LLM 模型名（默认 deepseek-chat 不支持）
os.environ.setdefault("LLM_MODEL", "deepseek-v4-flash")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from langchain_core.documents import Document
from qa_core.knowledge_graph.pipeline import run_knowledge_graph_pipeline
from qa_core.knowledge_graph.storage import GraphStorage, EMBEDDING_DIM


TEST_DOCS = [
    Document(
        page_content="""人事部发布了新的员工入职流程。根据规定，新员工需要在入职第一天前往HR部门办理手续，
领取工牌和办公设备。IT部门负责分配电脑和系统账号。财务部则负责办理工资卡和社保公积金。
此外，入职还需签署劳动合同、保密协议和员工手册确认书。""",
        metadata={"chunk_id": "chunk_001", "source": "hr"},
    ),
    Document(
        page_content="""财务部报销流程：员工出差前需填写出差申请单，经部门主管审批后生效。
出差结束后需在5个工作日内提交报销单，附上发票、行程单和住宿清单。
5000元以内的报销由部门主管审批，超过5000元需财务总监加签。""",
        metadata={"chunk_id": "chunk_002", "source": "finance"},
    ),
]


async def main():
    print("=" * 60)
    print("知识图谱端到端测试")
    print("=" * 60)

    # ── 1. 运行流水线 ──
    print("\n[1/4] 运行知识图谱构建管线...")
    result = await run_knowledge_graph_pipeline(
        chunks=TEST_DOCS,
        kb_version="test_kg_v1",
        max_gleanings=0,  # 不加 gleaning 节省 token
    )
    print(f"  处理 chunks: {result.processed_chunks}/{result.total_chunks}")
    print(f"  抽取实体: {result.entities_extracted}")
    print(f"  抽取关系: {result.relationships_extracted}")
    print(f"  社群数: {result.communities_detected}")
    print(f"  存储结果: {result.stored}")
    if result.errors:
        print(f"  错误: {result.errors}")

    if result.entities_extracted == 0:
        print("\n⚠ 未抽取到实体，跳过后续验证")
        return

    # ── 2. 检查 Milvus 实体集合 ──
    print("\n[2/4] 检查 Milvus 实体集合...")
    storage = GraphStorage()
    client = storage._get_client()
    if client.has_collection(storage._entity_collection):
        entities = client.query(
            collection_name=storage._entity_collection,
            limit=10,
            output_fields=["name", "type", "description"],
        )
        print(f"  实体集合有 {len(entities)} 条记录:")
        for e in entities:
            print(f"    [{e.get('type','?')}] {e.get('name','?'):12s} | {e.get('description','')[:50]}")
    else:
        print("  实体集合不存在")

    # ── 3. 检查 Milvus 关系集合 ──
    print("\n[3/4] 检查 Milvus 关系集合...")
    if client.has_collection(storage._relation_collection):
        relations = client.query(
            collection_name=storage._relation_collection,
            limit=10,
            output_fields=["source", "target", "description", "strength"],
        )
        print(f"  关系集合有 {len(relations)} 条记录:")
        for r in relations:
            print(f"    {r.get('source','?'):12s} → {r.get('target','?'):12s} | {r.get('description','')[:40]} (强度:{r.get('strength',0)})")
    else:
        print("  关系集合不存在")

    # ── 4. 清理测试数据 ──
    print("\n[4/4] 清理测试集合...")
    # storage.drop_collections()
    # print("  已删除 kg_entities, kg_relations, kg_communities")
    print("  保留集合供后续检查")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
