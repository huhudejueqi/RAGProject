"""FastAPI 对外请求/响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalDebugRequest(BaseModel):
    """HTTP 检索诊断接口的请求体。在线问答不复用该模型。

    调用顺序：上游业务入口 -> RetrievalDebugRequest。
    """

    query: str = Field(..., min_length=1)
    source_filter: str | None = None
    session_id: str | None = None
    scenario_id: str | None = None
    tenant_id: str | None = None
    dataset_id: str | None = None
    visibility: str | None = None
    user_role: str | None = None
    user_roles: list[str] = Field(default_factory=list)
    kb_version: str | None = None


class FeedbackRequest(BaseModel):
    """用户反馈载荷，rating 约束为 useful/not_useful。

    调用顺序：上游业务入口 -> FeedbackRequest。
    """

    session_id: str | None = None
    scenario_id: str | None = None
    tenant_id: str | None = None
    dataset_id: str | None = None
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    rating: str = Field(..., pattern="^(useful|not_useful)$")
    comment: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalDebugResponse(BaseModel):
    """检索调试响应，不包含最终答案，faq 和 doc 来源分开返回。

    调用顺序：上游业务入口 -> RetrievalDebugResponse。
    """

    query: str
    raw_query: str | None = None
    effective_query: str | None = None
    query_normalized: bool = False
    rewritten_query: str
    source_filter: str | None = None
    scenario_id: str | None = None
    tenant_id: str | None = None
    dataset_id: str | None = None
    visibility: str | None = None
    data_scope: dict[str, Any] | None = None
    kb_version: str | None = None
    route: str | None = None
    route_reason: str | None = None
    answer: str | None = None
    answer_confidence: dict[str, Any] = Field(default_factory=dict)
    intent: dict[str, Any]
    retrieval_plan: dict[str, Any] | None = None
    hit_type: str | None = None
    context_count: int | None = None
    faq_sources: list[dict[str, Any]] = Field(default_factory=list)
    doc_sources: list[dict[str, Any]] = Field(default_factory=list)


class AgentPlanRequest(BaseModel):
    """V2 Agent planning request.

    调用顺序：上游业务入口 -> AgentPlanRequest。
    """

    question: str = Field(..., min_length=1)


class AgentPlanResponse(BaseModel):
    """V2 Agent planning response.

    调用顺序：上游业务入口 -> AgentPlanResponse。
    """

    question: str
    plan: dict[str, Any]


class AgentRunRequest(BaseModel):
    """V2 Agent execution request.

    调用顺序：上游业务入口 -> AgentRunRequest。
    """

    question: str = Field(..., min_length=1)
    runtime: str = Field(default="deterministic", pattern="^(deterministic|langgraph|autonomous)$")
    use_strategy_drafts: bool = False


class UnifiedAskRequest(BaseModel):
    """Unified V1/V2 ask request for automatic routing.

    调用顺序：上游业务入口 -> UnifiedAskRequest。
    """

    question: str = Field(..., min_length=1)
    source_filter: str | None = None
    session_id: str | None = None
    scenario_id: str | None = None
    tenant_id: str | None = None
    dataset_id: str | None = None
    visibility: str | None = None
    user_role: str | None = None
    user_roles: list[str] = Field(default_factory=list)
    kb_version: str | None = None
    runtime: str = Field(default="autonomous", pattern="^(deterministic|langgraph|autonomous)$")
    prefer_mode: str = Field(default="auto", pattern="^(auto|knowledge|agent)$")
    use_strategy_drafts: bool = False


class UnifiedAskResponse(BaseModel):
    """Unified V1/V2 response with route decision.

    调用顺序：上游业务入口 -> UnifiedAskResponse。
    """

    route: dict[str, Any]
    knowledge: dict[str, Any] | None = None
    agent: dict[str, Any] | None = None


class AgentRunResponse(BaseModel):
    """V2 Agent execution response.

    调用顺序：上游业务入口 -> AgentRunResponse。
    """

    question: str
    strategy_meta: dict[str, Any] = Field(default_factory=dict)
    knowledge_answer: dict[str, Any] = Field(default_factory=dict)
    agentic_rag: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    business_closure: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any]
    execution: dict[str, Any]
    workbench: dict[str, Any]
    coordination: dict[str, Any]
    tools: list[dict[str, Any]] = Field(default_factory=list)


class StrategyDraftSaveRequest(BaseModel):
    """V2 strategy draft save request.

    调用顺序：上游业务入口 -> StrategyDraftSaveRequest。
    """

    drafts: dict[str, Any] = Field(default_factory=dict)


class StrategyDraftResponse(BaseModel):
    """V2 strategy draft response.

    调用顺序：上游业务入口 -> StrategyDraftResponse。
    """

    drafts: dict[str, Any] = Field(default_factory=dict)
    saved_path: str | None = None


class EnterpriseEvalRunRequest(BaseModel):
    """V2 enterprise evaluation request.

    调用顺序：上游业务入口 -> EnterpriseEvalRunRequest。
    """

    runtime: str = Field(default="langgraph", pattern="^(deterministic|langgraph|autonomous)$")


class EnterpriseBackupRequest(BaseModel):
    """V2 enterprise backup request.

    调用顺序：上游业务入口 -> EnterpriseBackupRequest。
    """

    description: str = ""


class EnterpriseDrDrillRequest(BaseModel):
    """Run a V2.1 disaster recovery drill.

    调用顺序：上游业务入口 -> EnterpriseDrDrillRequest。
    """

    description: str = "v2.1 disaster recovery drill"
    apply: bool = False


class EnterpriseRestoreRequest(BaseModel):
    """V2 enterprise restore request.

    调用顺序：上游业务入口 -> EnterpriseRestoreRequest。
    """

    backup_id: str = Field(..., min_length=1)
    confirm: str | None = None


class EnterpriseApprovalCreateRequest(BaseModel):
    """Create a human-in-the-loop approval item.

    调用顺序：上游业务入口 -> EnterpriseApprovalCreateRequest。
    """

    task: dict[str, Any] = Field(default_factory=dict)


class EnterpriseApprovalDecisionRequest(BaseModel):
    """Decide a human-in-the-loop approval item.

    调用顺序：上游业务入口 -> EnterpriseApprovalDecisionRequest。
    """

    approval_id: str = Field(..., min_length=1)
    decision: str = Field(..., pattern="^(approved|rejected)$")


class EnterprisePermissionCheckRequest(BaseModel):
    """Check RBAC permission for one role/action pair.

    调用顺序：上游业务入口 -> EnterprisePermissionCheckRequest。
    """

    role: str = "viewer"
    action: str = Field(..., min_length=1)


class EnterpriseProtocolInvokeRequest(BaseModel):
    """Invoke a V2 MCP/A2A compatible protocol action.

    调用顺序：上游业务入口 -> EnterpriseProtocolInvokeRequest。
    """

    protocol: str = Field(default="mcp", pattern="^(mcp|a2a)$")
    tool_name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    question: str | None = None
    runtime: str = Field(default="deterministic", pattern="^(deterministic|langgraph|autonomous)$")


class ExecutionTaskCreateRequest(BaseModel):
    """Create an executable task from one V2 run.

    调用顺序：上游业务入口 -> ExecutionTaskCreateRequest。
    """

    run_id: str = Field(..., min_length=1)


class ExecutionTaskActionRequest(BaseModel):
    """Mutate one V2 execution task.

    调用顺序：上游业务入口 -> ExecutionTaskActionRequest。
    """

    action: str = Field(..., pattern="^(approve|reject|execute|retry|cancel)$")


class QueueTaskCreateRequest(BaseModel):
    """Create a V2.1 queue task from an execution task.

    调用顺序：上游业务入口 -> QueueTaskCreateRequest。
    """

    execution_task_id: str = Field(..., min_length=1)
    run_id: str = ""
    title: str = ""
    priority: int = Field(default=5, ge=1, le=10)
    timeout_sec: int = Field(default=300, ge=10, le=86400)
    payload: dict[str, Any] = Field(default_factory=dict)


class QueueTaskActionRequest(BaseModel):
    """Mutate a V2.1 queue task.

    调用顺序：上游业务入口 -> QueueTaskActionRequest。
    """

    action: str = Field(..., pattern="^(pause|resume|cancel|retry|worker_once)$")


class ConnectorInvokeRequest(BaseModel):
    """Invoke or preview one V2 external connector.

    调用顺序：上游业务入口 -> ConnectorInvokeRequest。
    """

    connector_name: str = Field(default="webhook_ticket", min_length=1)
    mode: str = Field(default="dry_run", pattern="^(dry_run|apply)$")
    endpoint_url: str | None = None
    method: str = Field(default="POST", pattern="^(POST|PUT|PATCH)$")
    headers: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    timeout_sec: float = Field(default=5.0, ge=1, le=30)
    retry_count: int = Field(default=0, ge=0, le=2)
    confirm: str | None = None


class GraphRAGRebuildRequest(BaseModel):
    """Rebuild the V2.1 GraphRAG index.

    调用顺序：上游业务入口 -> GraphRAGRebuildRequest。
    """

    scenario_id: str | None = None
    max_files: int = Field(default=200, ge=1, le=2000)


class GraphRAGQueryRequest(BaseModel):
    """Query the V2.1 GraphRAG index.

    调用顺序：上游业务入口 -> GraphRAGQueryRequest。
    """

    question: str = Field(..., min_length=1)
    scenario_id: str | None = None
    max_paths: int = Field(default=5, ge=1, le=20)
