"""MySQL 存储基类：延迟创建 SQLAlchemy 引擎。"""
from __future__ import annotations

import re

from sqlalchemy import create_engine

from qa_core.config.settings import get_settings


_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_sql_identifier(value: str, *, label: str = "SQL identifier") -> str:
    """校验可拼接到 SQL 中的表名/索引名，避免配置项注入 SQL。

    调用顺序：问答历史或反馈存储 -> safe_sql_identifier()。
    """
    if not _SQL_IDENTIFIER_RE.fullmatch(value or ""):
        raise ValueError(f"{label} 不合法：{value!r}")
    return value

class _MySqlStore:
    """MySQL 存储的轻量基类。

    基类统一加载项目配置并延迟创建 SQLAlchemy 引擎，子类只需要关心自己的表名和业务方法。

    调用顺序：问答历史或反馈存储 -> _MySqlStore。
    """

    def __init__(self) -> None:
        """初始化 MySQL 存储基类：加载全局配置并延迟创建引擎（首次访问 engine 属性时才连接数据库）。

        调用顺序：问答历史或反馈存储 -> _MySqlStore.__init__()。
        """
        self.settings = get_settings()
        self._engine = None

    @property
    def engine(self):
        """延迟创建带连接健康检查的 SQLAlchemy 同步引擎。

        调用顺序：问答历史或反馈存储 -> _MySqlStore.engine()。
        """
        if self._engine is None:
            # 延迟创建 SQLAlchemy 引擎（带连接健康检查）
            self._engine = create_engine(self.settings.mysql_sync_uri, pool_pre_ping=True)
        return self._engine
