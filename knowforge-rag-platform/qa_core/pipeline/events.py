"""WebSocket 流式问答事件构造器：start / status / token / end / error 五种事件格式集中管理。
"""

from __future__ import annotations
import time
from typing import Any


def user_facing_error_message(error: BaseException | str) -> str:
    """将内部异常转换为可展示给用户的稳定提示。

    供应商 SDK 的错误文本可能包含账户状态、请求参数和内部标识，不能直接
    通过 WebSocket 返回。完整异常仍由调用方写入日志和 Trace。
    """
    text = str(error).lower()
    if any(
        marker in text
        for marker in (
            "arrearage",
            "overdue-payment",
            "access denied",
            "unauthorized",
            "authentication",
            "api key",
            "invalid_api_key",
            "invalid api key",
            "401",
            "403",
        )
    ):
        return "当前生成服务暂不可用，请检查模型服务账户或 API Key，并联系管理员。"
    if any(marker in text for marker in ("timeout", "timed out", "time out")):
        return "模型服务响应超时，请稍后重试。"
    if any(marker in text for marker in ("connection", "connecterror", "connection refused")):
        return "模型服务暂时无法连接，请稍后重试。"
    return "抱歉，处理失败，请稍后重试。"

def start_event(
    *,
    session_id: str,
    trace_id: str,
    scenario_id: str,
    scenario_name: str,
    data_scope: dict[str, Any],
    kb_version: str | None,
) -> dict[str, Any]:
    """通知前端 WebSocket 连接已建立、请求已接收，前端据此展示加载状态。

    调用顺序：QAService/RAG 管线 -> start_event()。
    """
    return {
        "type": "start",
        "session_id": session_id,
        "trace_id": trace_id,
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "data_scope": data_scope,
        "kb_version": kb_version,
    }


def status_event(message: str, session_id: str) -> dict[str, Any]:
    """让前端展示当前处理阶段（意图识别/检索/生成），缓解用户等待焦虑。

    调用顺序：QAService/RAG 管线 -> status_event()。
    """
    return {"type": "status", "message": message, "session_id": session_id}


def token_event(token: str, session_id: str) -> dict[str, Any]:
    """流式逐字推送生成结果，让用户逐步看到内容而非等待完整响应。

    调用顺序：QAService/RAG 管线 -> token_event()。
    """
    return {"type": "token", "token": token, "session_id": session_id}


def end_event(
    *,
    session_id: str,
    hit_type: str,
    sources: list[dict[str, Any]],
    started: float,
    rewritten_query: str | None,
    trace_id: str,
    intent: dict[str, Any],
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    """通知前端生成结束以关闭加载状态，同时附带耗时/来源等诊断数据供后续分析。

    调用顺序：QAService/RAG 管线 -> end_event()。
    """
    return {
        "type": "end",
        "session_id": session_id,
        "trace_id": trace_id,
        "is_complete": True,
        "hit_type": hit_type,
        "sources": sources,
        "answer_confidence": retrieval["answer_confidence"],
        "rewritten_query": rewritten_query,
        "intent": intent,
        "retrieval": retrieval,
        "stage_timings_ms": retrieval["stage_timings_ms"],
        "first_token_ms": retrieval["first_token_ms"],
        "slowest_stage": retrieval["slowest_stage"],
        "processing_time": time.perf_counter() - started,
    }


def error_event(*, error: str, session_id: str, trace_id: str) -> dict[str, Any]:
    """以事件形式将异常推送给前端，避免 WebSocket 断开，前端可展示友好提示。

    调用顺序：QAService/RAG 管线 -> error_event()。
    """
    return {"type": "error", "error": error, "session_id": session_id, "trace_id": trace_id}
