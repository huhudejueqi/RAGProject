"""运行期 MySQL schema bootstrap。

API 启动、入库脚本和版本运维脚本在业务读写前显式调用这里；业务 Store 不再在方法
内部夹带 DDL 副作用。
"""

from __future__ import annotations

from sqlalchemy import create_engine

from qa_core.config.settings import get_settings
from qa_core.cache.namespaces import CACHE_NAMESPACE_TABLE
from qa_core.governance.chunk_versions import KB_CHUNK_VERSIONS_TABLE
from qa_core.governance.kb_versions import KB_ACTIVATION_TABLE, KB_ACTIVE_TABLE, KB_VERSIONS_TABLE
from qa_core.indexing.manifest import INDEX_MANIFEST_TABLE
from qa_core.memory.base import safe_sql_identifier
from qa_core.storage.mysql_schema import RUNTIME_SCHEMA_SQL, run_runtime_schema_sql


DEFAULT_CHAT_MESSAGES_TABLE = "chat_messages"


def bootstrap_mysql_schema() -> dict[str, object]:
    """初始化运行期 MySQL 表结构，并返回本次处理摘要。

    调用顺序：启动期 schema bootstrap -> bootstrap_mysql_schema()。
    """
    settings = get_settings()
    # pool_pre_ping=True 确保每次取连接前发送轻量 SELECT 1 探测，避免复用失效连接导致 2006 错误
    engine = create_engine(settings.mysql_sync_uri, pool_pre_ping=True)

    # 对所有运行时表名做 SQL 防注入标识符转义（反引号包裹），表名来自配置或常量而非用户输入
    version_table = safe_sql_identifier(KB_VERSIONS_TABLE, label="KB versions table")
    active_table = safe_sql_identifier(KB_ACTIVE_TABLE, label="KB active table")
    activation_table = safe_sql_identifier(KB_ACTIVATION_TABLE, label="KB activation table")
    cache_namespace_table = safe_sql_identifier(CACHE_NAMESPACE_TABLE, label="Cache namespace table")
    chunk_version_table = safe_sql_identifier(KB_CHUNK_VERSIONS_TABLE, label="KB chunk versions table")
    manifest_table = safe_sql_identifier(INDEX_MANIFEST_TABLE, label="Index manifest table")
    feedback_table = safe_sql_identifier(settings.feedback_table_name, label="Feedback table")
    summary_table = safe_sql_identifier(settings.chat_summary_table_name, label="Chat summary table")
    chat_messages_table = safe_sql_identifier(DEFAULT_CHAT_MESSAGES_TABLE, label="Chat messages table")

    # 将替换后的表名注入 SQL 模板并逐条执行，确保所有运行时表在业务读写前完成 DDL 建表
    statement_count = run_runtime_schema_sql(
        engine,
        {
            "KB_VERSIONS_TABLE": version_table,
            "KB_ACTIVE_TABLE": active_table,
            "KB_ACTIVATION_TABLE": activation_table,
            "CACHE_NAMESPACE_TABLE": cache_namespace_table,
            "KB_CHUNK_VERSIONS_TABLE": chunk_version_table,
            "INDEX_MANIFEST_TABLE": manifest_table,
            "FEEDBACK_TABLE": feedback_table,
            "CHAT_SUMMARY_TABLE": summary_table,
            "CHAT_MESSAGES_TABLE": chat_messages_table,
        },
    )

    return {
        "mysql_database": settings.mysql_database,
        "tables": [
            version_table,
            active_table,
            activation_table,
            cache_namespace_table,
            chunk_version_table,
            manifest_table,
            feedback_table,
            summary_table,
            chat_messages_table,
        ],
        "schema_sql": str(RUNTIME_SCHEMA_SQL),
        "statement_count": statement_count,
    }
