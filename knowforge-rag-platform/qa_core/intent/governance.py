"""Intent model governance helpers for V1 release checks and admin status."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qa_core.common import path_updated_at, read_json_dict, utc_now, write_json
from qa_core.config.settings import PROJECT_ROOT, get_settings
from qa_core.intent.decision import POLICY
from qa_core.intent.model_classifier import (
    DEFAULT_EVAL_EXAMPLES,
    LABELS_FILENAME,
    RETRIEVAL_INTENTS,
    BertIntentModelService,
)


INTENT_MODEL_REPORT_DIR = PROJECT_ROOT / "reports" / "intent_model"
INTENT_MODEL_LATEST_REPORT = INTENT_MODEL_REPORT_DIR / "intent_model_latest.json"


def build_intent_model_report(*, evaluate: bool = True) -> dict[str, Any]:
    """Build a deterministic governance report for the local intent model."""

    settings = get_settings()
    model_path = Path(settings.intent_model_path)
    labels_payload = read_json_dict(model_path / LABELS_FILENAME)
    required_files = ["config.json", LABELS_FILENAME, "tokenizer.json", "vocab.txt"]
    missing_files = [name for name in required_files if not (model_path / name).exists()]
    has_weights = (model_path / "model.safetensors").exists() or (model_path / "pytorch_model.bin").exists()
    artifact_ok = model_path.exists() and not missing_files and has_weights

    evaluation_payload: dict[str, Any] = {}
    warmup_payload: dict[str, Any] = {}
    runtime_ok = artifact_ok
    error = ""
    if artifact_ok:
        try:
            service = BertIntentModelService.from_settings()
            prediction = service.predict("新人入职流程有哪些", has_history=False)
            warmup_payload = {
                "model_version": service.model_version,
                "labels": list(service.labels),
                "sample_intent": prediction.intent,
                "sample_score": round(prediction.score, 4),
                "policy_version": POLICY.policy_version,
            }
            if evaluate:
                evaluation_payload = service.evaluate(DEFAULT_EVAL_EXAMPLES).as_dict()
            runtime_ok = True
        except Exception as exc:  # pragma: no cover - exercised by deployment checks
            runtime_ok = False
            error = str(exc)

    accuracy = float(evaluation_payload.get("accuracy") or 0.0)
    labels = labels_payload.get("labels") if isinstance(labels_payload.get("labels"), list) else []
    ok = bool(artifact_ok and runtime_ok and tuple(labels) == RETRIEVAL_INTENTS and (not evaluate or accuracy >= 0.75))
    return {
        "report_type": "intent_model_governance",
        "created_at": utc_now(),
        "ok": ok,
        "artifact_ok": artifact_ok,
        "runtime_ok": runtime_ok,
        "error": error,
        "model": {
            "model_path": str(model_path),
            "model_version": settings.intent_model_version,
            "device": settings.intent_model_device,
            "max_length": settings.intent_model_max_length,
            "labels": labels,
            "expected_labels": list(RETRIEVAL_INTENTS),
            "base_model": labels_payload.get("base_model"),
            "training_examples": labels_payload.get("training_examples"),
            "eval_examples": labels_payload.get("eval_examples") or len(DEFAULT_EVAL_EXAMPLES),
            "updated_at": path_updated_at(model_path / LABELS_FILENAME) if (model_path / LABELS_FILENAME).exists() else "",
        },
        "artifact": {
            "required_files": required_files,
            "missing_files": missing_files,
            "has_weights": has_weights,
        },
        "evaluation": evaluation_payload,
        "warmup": warmup_payload,
        "decision_policy": {
            "policy_version": POLICY.policy_version,
            "model_min_score": POLICY.model_min_score,
            "agreement_score_boost": POLICY.agreement_score_boost,
            "conflict_final_score": POLICY.conflict_final_score,
        },
        "closure": {
            "online_gateway": "qa_core.intent.decision.apply_intent_decision_gateway",
            "training_script": "scripts/train_intent_bert.py",
            "model_eval_script": "scripts/demo_intent_model.py --eval-only",
            "policy_eval_script": "scripts/evaluate_intent_policy.py --fail-on-critical",
            "admin_endpoint": "/api/admin/intent_model",
            "latest_report": str(INTENT_MODEL_LATEST_REPORT.relative_to(PROJECT_ROOT)),
        },
    }


def write_intent_model_report(path: str | Path = INTENT_MODEL_LATEST_REPORT, *, evaluate: bool = True) -> str:
    """Write the latest intent-model governance report."""

    return write_json(path, build_intent_model_report(evaluate=evaluate))


def latest_intent_model_report() -> dict[str, Any]:
    """Return latest report if present, otherwise build a live read-only report."""

    payload = read_json_dict(INTENT_MODEL_LATEST_REPORT)
    if payload:
        return {
            "available": True,
            "file": str(INTENT_MODEL_LATEST_REPORT.relative_to(PROJECT_ROOT)),
            "updated_at": path_updated_at(INTENT_MODEL_LATEST_REPORT),
            "payload": payload,
        }
    return {
        "available": False,
        "file": None,
        "payload": build_intent_model_report(evaluate=False),
    }
