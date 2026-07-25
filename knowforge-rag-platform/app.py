"""KnowForge RAG Platform 的 FastAPI 应用入口。

本文件现在只负责四件事：
1. 创建 FastAPI 应用；
2. 配置 CORS 和静态资源；
3. 启动时执行必需环境校验、MySQL schema bootstrap、active 版本校验和模型预热；
4. 注册 `qa_core.api` 下拆分后的 V1 路由。

为什么要保持入口文件很薄：
- `app.py` 是服务启动点，不应该继续堆 HTTP、WebSocket、管理接口和 RAG 细节；
- 接口按页面、聊天、管理、知识库版本拆分，入口保持薄封装；
- 入口越薄，越容易确认当前项目没有旧链路、没有技术降级路径、没有隐藏旁路。

不适合放在这里的内容：
- 不要在这里实现意图识别、检索策略、提示词拼接或 Milvus 查询；
- 不要在这里保存用户会话状态；
"""

from __future__ import annotations
import asyncio
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from qa_core.api import admin, chat, kb_versions, pages
from qa_core.api.error_handlers import register_api_exception_handlers
from qa_core.config.logging_config import get_logger
from qa_core.config.preflight import validate_active_kb_versions, validate_runtime_environment
from qa_core.config.settings import get_settings
from qa_core.intent.decision import warmup_intent_decision_gateway
from qa_core.llm.client import refresh_llm_runtime_status
from qa_core.observability.langsmith_adapter import configure_langsmith_environment
from qa_core.retrieval.factory import start_retrieval_warmup_background
from qa_core.storage.bootstrap import bootstrap_mysql_schema

settings = get_settings()
configure_langsmith_environment()
logger = get_logger(__name__)
app = FastAPI(
    title="KnowForge RAG Platform API",
    description="LangChain + Milvus Hybrid 企业级多场景 RAG 知识平台",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)
register_api_exception_handlers(app)

# 当前前端和 API 默认同源部署，但保留 CORS 配置是为了方便本地调试：
# 例如单独启动 Vite/React 页面时，只需要在当前运行配置中追加允许来源即可。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


async def refresh_llm_status_background() -> None:
    """后台刷新 LLM 连通性状态，避免供应商接口慢响应拖慢应用启动。"""
    llm_summary = await asyncio.to_thread(refresh_llm_runtime_status)
    logger.info("Runtime LLM connectivity status: %s", llm_summary)


@app.on_event("startup")
async def warmup_runtime() -> None:
    """服务启动时执行前置校验、schema bootstrap，并启动后台检索栈预热。

    当前项目的基础环境是必需前置条件：LLM Key、Milvus、MySQL、本地模型、场景配置和
    active 知识库版本缺失时，服务直接启动失败。LLM 供应商真实连通性在启动期探测并
    写入运行状态，不阻断页面和治理接口启动。

    调用顺序：FastAPI 启动入口 -> warmup_runtime()。
    """
    summary = validate_runtime_environment()
    logger.info("Runtime preflight passed: %s", summary)
    schema_summary = await asyncio.to_thread(bootstrap_mysql_schema)
    logger.info("Runtime MySQL schema bootstrap passed: %s", schema_summary)
    active_summary = validate_active_kb_versions(settings.active_scenario_id)
    logger.info("Runtime active KB version check passed: %s", active_summary)
    intent_summary = warmup_intent_decision_gateway()
    logger.info("Runtime BERT intent model warmup passed: %s", intent_summary)
    asyncio.create_task(refresh_llm_status_background())
    logger.info("Runtime LLM connectivity probe started")
    retrieval_summary = start_retrieval_warmup_background()
    logger.info("Runtime retrieval warmup started: %s", retrieval_summary)

app.include_router(pages.router)
app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(kb_versions.router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
