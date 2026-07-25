"""知识库版本管理路由。只管理本地版本清单，不直接删除 Milvus 数据。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from qa_core.api.dependencies import require_admin_token
from qa_core.api.error_handlers import raise_bad_request, raise_not_found
from qa_core.governance.kb_versions import get_kb_version_store
router = APIRouter()


class ActivateVersionRequest(BaseModel):
    """Optional audit metadata for a version activation request.

    调用顺序：FastAPI 路由层 -> ActivateVersionRequest。
    """

    reason: str = ""
    activated_by: str = "admin"


@router.get("/api/kb_versions")
async def list_kb_versions(scenario_id: str | None = None):
    """返回知识库版本清单和当前生效版本。

    调用顺序：FastAPI 路由层 -> list_kb_versions()。
    """
    return get_kb_version_store(scenario_id).as_payload()

@router.post("/api/kb_versions/{kb_version}/activate")
async def activate_kb_version(
    kb_version: str,
    scenario_id: str | None = None,
    payload: ActivateVersionRequest | None = None,
    _: None = Depends(require_admin_token),
):
    """把已发布过的历史版本切回 active。

    调用顺序：FastAPI 路由层 -> activate_kb_version()。
    """
    request_payload = payload or ActivateVersionRequest()
    store = get_kb_version_store(scenario_id)
    record = store.get(kb_version)
    if record is None:
        raise_not_found(f"知识库版本不存在：{kb_version}")
    if record.status == "ARCHIVED":
        raise_bad_request("归档版本不能直接激活")
    if not record.activated_at:
        raise_bad_request("新版本发布必须通过 scripts/rebuild_kb_version.py --quality-gate --activate；管理接口只用于回滚已发布版本")
    try:
        version = store.activate_version(
            kb_version,
            reason=request_payload.reason or "admin_rollback",
            activated_by=request_payload.activated_by or "admin",
        )
        return {"status": "success", "version": version.as_dict()}
    except ValueError as exc:
        raise_not_found(str(exc))

@router.post("/api/kb_versions/{kb_version}/archive")
async def archive_kb_version(kb_version: str, scenario_id: str | None = None, _: None = Depends(require_admin_token)):
    """归档一个非 active 知识库版本。

    调用顺序：FastAPI 路由层 -> archive_kb_version()。
    """
    try:
        version = get_kb_version_store(scenario_id).archive_version(kb_version)
        return {"status": "success", "version": version.as_dict()}
    except ValueError as exc:
        raise_bad_request(str(exc))
