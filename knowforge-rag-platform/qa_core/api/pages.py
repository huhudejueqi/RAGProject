"""页面、健康检查和会话创建路由。不参与 RAG 检索和答案生成。"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse

from qa_core.llm.client import llm_runtime_status
from qa_core.retrieval.factory import retrieval_warmup_state
from qa_core.scenarios.registry import resolve_scenario


router = APIRouter()

def _static_page(path: str):
    """返回一个禁止浏览器缓存的静态文件响应。

    调用顺序：FastAPI 路由层 -> _static_page()。
    """
    return FileResponse(path, headers={"Cache-Control": "no-store"})

@router.get("/")
async def read_root():
    """提供单页聊天界面，并禁止浏览器缓存旧版 JS。

    调用顺序：FastAPI 路由层 -> read_root()。
    """
    return _static_page("static/index.html")

@router.get("/admin")
async def read_admin_page():
    """提供本地状态页。

    调用顺序：FastAPI 路由层 -> read_admin_page()。
    """
    return _static_page("static/admin.html")

@router.get("/health")
async def health_check():
    """容器与本地健康检查接口。

    调用顺序：FastAPI 路由层 -> health_check()。
    """
    scenario = resolve_scenario()
    return {
        "status": "healthy",
        "engine": "langchain_milvus_hybrid",
        "active_scenario_id": scenario.scenario_id,
        "active_scenario_name": scenario.display_name,
        "llm": llm_runtime_status(),
        "retrieval_warmup": retrieval_warmup_state(),
    }

@router.post("/api/create_session")
async def create_session(scenario_id: str | None = None):
    """创建页面端使用的会话 ID。

    调用顺序：FastAPI 路由层 -> create_session()。
    """
    scenario = resolve_scenario(scenario_id)
    return {"session_id": f"{scenario.scenario_id}:{uuid.uuid4()}", "scenario_id": scenario.scenario_id}

@router.get("/docs")
async def redirect_docs_root():
    """把文档首页重定向到带尾斜杠的地址，保证 MkDocs 相对资源路径正确。

    调用顺序：FastAPI 路由层 -> redirect_docs_root() -> read_docs()。
    """
    return RedirectResponse(url="/docs/", status_code=307)

@router.get("/docs/{full_path:path}")
async def read_docs(full_path: str = ""):
    """提供 mkdocs 构建的文档页面（site/ 目录）。

    调用顺序：FastAPI 路由层 -> read_docs()。
    """
    docs_dir = Path("site")
    if not full_path or full_path.endswith("/"):
        full_path = os.path.join(full_path, "index.html")
    elif not full_path.endswith(".html") and "." not in Path(full_path).suffix:
        full_path = full_path + ".html" if not full_path.endswith("/") else os.path.join(full_path, "index.html")

    file_path = docs_dir / full_path
    if not file_path.exists() or not file_path.is_file():
        file_path = docs_dir / "index.html"

    return FileResponse(str(file_path), headers={"Cache-Control": "no-store"})
