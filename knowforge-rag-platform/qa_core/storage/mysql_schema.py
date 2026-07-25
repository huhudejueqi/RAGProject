"""MySQL schema SQL runner.

The table DDL lives in ``runtime_schema.sql``. This module only renders table
name placeholders and executes that file during startup or script bootstrap.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

from sqlalchemy import text


RUNTIME_SCHEMA_SQL = Path(__file__).with_name("runtime_schema.sql")
_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def run_runtime_schema_sql(engine, variables: Mapping[str, str]) -> int:
    """执行 runtime_schema.sql 并返回实际执行的 SQL 语句数量。

    调用顺序：bootstrap_mysql_schema() -> run_runtime_schema_sql()
    -> _render_sql() -> _split_sql_statements() -> SQLAlchemy 执行。
    """
    if not RUNTIME_SCHEMA_SQL.exists():
        # SQL 模板文件丢失时直接阻断，避免服务启动后缺少核心表导致业务层报"表不存在"
        raise RuntimeError(f"MySQL runtime schema SQL file is missing: {RUNTIME_SCHEMA_SQL}")

    # 步骤1：将 SQL 模板中的 {{TABLE_NAME}} 占位符替换为运行时确定的真实表名
    rendered = _render_sql(RUNTIME_SCHEMA_SQL.read_text(encoding="utf-8"), variables)
    # 步骤2：去除行注释并按分号拆分，得到可逐条执行的 SQL 语句列表
    statements = _split_sql_statements(rendered)
    # 步骤3：在事务中逐条执行 DDL，engine.begin() 自动提交事务，任一语句失败会整体回滚
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
    return len(statements)


def _render_sql(source: str, variables: Mapping[str, str]) -> str:
    """渲染 SQL 模板中的表名占位符。

    调用顺序：run_runtime_schema_sql() -> _render_sql() -> replace()。
    """

    def replace(match: re.Match[str]) -> str:
        """替换 SQL 模板中的占位符（{{NAME}}）为实际值。

        调用顺序：启动期 schema bootstrap -> replace()。
        """
        name = match.group(1)
        if name not in variables:
            # 模板中出现了调用方未提供的占位符，阻止生成不完整的 DDL，避免漏建表
            raise RuntimeError(f"MySQL schema placeholder is not provided: {name}")
        return variables[name]

    # 全局搜索 {{VARIABLE_NAME}} 模式，每匹配到一处就回调 replace 执行替换
    return _PLACEHOLDER_RE.sub(replace, source)


def _split_sql_statements(sql: str) -> list[str]:
    """将 SQL 文件内容拆成可逐条执行的语句。

    调用顺序：run_runtime_schema_sql() -> _split_sql_statements()
    -> _strip_line_comments()。
    """
    content = _strip_line_comments(sql)
    statements: list[str] = []
    current: list[str] = []
    in_single_quote = False
    in_double_quote = False
    escape_next = False

    # 逐字符扫描，区分引号内分号和引号外分号，避免误将字符串字面量中的分号当作语句分隔
    for char in content:
        current.append(char)
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if char == ";" and not in_single_quote and not in_double_quote:
            # 遇到引号外的分号时截断一条完整语句，去尾分号后加入结果列表
            statement = "".join(current).strip().rstrip(";").strip()
            if statement:
                statements.append(statement)
            current = []

    # 文件末尾可能无尾部分号，剩余内容也是一条完整语句
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def _strip_line_comments(sql: str) -> str:
    """移除 SQL 文件中的空行和行注释。

    调用顺序：_split_sql_statements() -> _strip_line_comments()。
    """
    lines = []
    for line in sql.splitlines():
        stripped = line.strip()
        # 过滤 `--` 开头的行注释和空行，保留实际 DDL 语句行
        if not stripped or stripped.startswith("--"):
            continue
        lines.append(line)
    return "\n".join(lines)
