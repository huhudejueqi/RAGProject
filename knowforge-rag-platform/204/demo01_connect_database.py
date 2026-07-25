"""
演示 pymilvus 提供的两种连接方式及其适用场景：
  方式一：connections 模块（底层，本项目 LangChain 兼容层使用）
  方式二：MilvusClient 高层 API（新版推荐，更简洁）
    在pymilvus模块包下的推荐使用方式

需要注意：现在使用的是pymilvus而不是langchain与milvus的集成包，pymilvus提供了操作milvus的很多底层支持
学习要点：
  1. 理解两种连接方式的设计差异
  2. 知道项目中为什么同时存在两种连接
  3. 能独立创建 MilvusClient 并查看 Database 和 Collection 列表

Attu：客户端
pycharm：客户端
虚拟机->docker->milvus：服务端
wsl->docker->milvus：服务端
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# connections 模块：pymilvus 底层的 ORM 风格连接管理
#   - 通过 alias（别名）标识不同连接
#   - 全局连接注册表，所有组件共享
#   - LangChain 的 Milvus wrapper 底层依赖此模块
# ---------------------------------------------------------------------------
from pymilvus import connections
from milvus_common import MILVUS_DB, MILVUS_URI, connect_client

def main():
    # ========================================================================
    # 方式一：connections 模块 — 底层 ORM 风格连接
    # ========================================================================
    # 这是 pymilvus 的传统连接方式。核心概念是 alias（别名）：
    #   - 每个连接有一个唯一 alias，后续操作通过 alias 引用该连接
    #   - 所有连接注册在全局的 connections._connected 字典中
    #   - 本项目的 langchain-milvus → PyMilvus 兼容层（milvus_compat.py）使用此方式
    #
    # 为什么项目要用 connections.connect 而不是 MilvusClient？
    #   → LangChain 的 Milvus 封装内部通过 connections 模块管理连接，
    #     而不是通过 MilvusClient。为了兼容 LangChain 的 hybrid search 路径，
    #     项目需要手动在 connections 注册表中登记 alias。
    print("=== 方式一：connections 模块，本项目 LangChain 兼容层常用 ===")
    # db_name：Milvus 2.4+ 支持多 Database，指定要连接的具体 Database
    connections.connect(alias="lecture04", uri=MILVUS_URI, db_name=MILVUS_DB)
    print("Connected alias: lecture04")

    # ========================================================================
    # 方式二：MilvusClient — 新版高层 API（推荐用于脚本和教学）
    # ========================================================================
    # MilvusClient 是 pymilvus 2.4+ 推荐的使用方式：
    #   - 内部自动管理 gRPC 连接池（无需手动管理连接生命周期）
    #   - 提供统一的 create_collection / insert / search / delete 接口
    #   - 支持上下文管理器（with 语句）
    #   - 线程安全
    #
    # 教学 demo 统一使用 MilvusClient，因为：
    #   1. 代码更简洁
    #   2. 不需要手动管理 alias
    #   3. 与 Milvus 官方文档风格一致
    print("\n=== 方式二：MilvusClient，新版 API 更适合教学和脚本 ===")
    client = connect_client()  # 内部调用 MilvusClient(uri=...)，并切换到 MILVUS_DB
    print(f"URI: {MILVUS_URI}")
    print(f"Database: {MILVUS_DB}")
    # list_databases()：列出 Milvus 实例中所有 Database
    print(f"Databases: {client.list_databases()}")
    # list_collections()：列出当前 Database 中所有 Collection
    print(f"Collections: {client.list_collections()}")


if __name__ == "__main__":
    main()
