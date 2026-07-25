"""BERT fine-tuned intent model service used by the V1 intent gateway.

The online RAG chain calls this module after deterministic routing and before
building the retrieval plan. It loads a local HuggingFace
``BertForSequenceClassification`` artifact and returns a governed prediction
payload for ``FAQ_QUERY / KNOWLEDGE_QUERY / FOLLOW_UP``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


RETRIEVAL_INTENTS = ("FAQ_QUERY", "KNOWLEDGE_QUERY", "FOLLOW_UP")
LABELS_FILENAME = "intent_labels.json"


@dataclass(frozen=True)
class IntentTrainingExample:
    """Single supervised sample for BERT intent fine-tuning."""

    query: str
    label: str
    has_history: bool = False


@dataclass(frozen=True)
class IntentModelPrediction:
    """Model prediction returned to the intent decision gateway."""

    intent: str
    score: float
    scores: dict[str, float]
    reason: str
    model_version: str

    def as_dict(self) -> dict[str, object]:
        """转换为可 JSON 序列化的诊断数据。

        调用顺序：测试或业务入口 -> IntentModelPrediction.as_dict()。
        """
        return {
            "intent": self.intent,
            "score": round(self.score, 4),
            "scores": {intent: round(score, 4) for intent, score in self.scores.items()},
            "reason": self.reason,
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class IntentModelEvaluation:
    """Offline evaluation result for a labeled validation set."""

    accuracy: float
    total: int
    correct: int
    confusion_matrix: dict[str, dict[str, int]]

    def as_dict(self) -> dict[str, object]:
        """转换为可 JSON 序列化的诊断数据。

        调用顺序：测试或业务入口 -> IntentModelEvaluation.as_dict()。
        """
        return {
            "accuracy": round(self.accuracy, 4),
            "total": self.total,
            "correct": self.correct,
            "confusion_matrix": self.confusion_matrix,
        }


def format_intent_model_input(query: str, *, has_history: bool) -> str:
    """Format the classifier input consistently for training and inference."""

    history_marker = "有历史对话" if has_history else "无历史对话"
    return f"{history_marker}。用户问题：{query.strip()}"


class BertIntentModelService:
    """Local BERT sequence-classification service for retrieval intents."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        max_length: int = 64,
        model_version: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.device = _resolve_device(device)
        _require_model_artifact(self.model_path)
        self.labels = _load_label_sequence(self.model_path)
        self.label2id = {label: index for index, label in enumerate(self.labels)}
        self.model_version = model_version or _read_model_version(self.model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        if int(self.model.config.num_labels) != len(self.labels):
            raise RuntimeError(
                f"意图模型标签数量不一致：model num_labels={self.model.config.num_labels}, "
                f"{LABELS_FILENAME} labels={len(self.labels)}"
            )
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_settings(cls) -> "BertIntentModelService":
        """Build the service from runtime settings."""

        from qa_core.config.settings import get_settings

        settings = get_settings()
        return cls(
            settings.intent_model_path,
            device=settings.intent_model_device,
            max_length=settings.intent_model_max_length,
            model_version=settings.intent_model_version,
        )

    def predict(self, query: str, *, has_history: bool = False) -> IntentModelPrediction:
        """Predict retrieval intent with the local fine-tuned BERT model.

        执行流程：
        1. 将用户查询格式化为模型输入文本（含对话历史标记）
        2. Tokenize → 截断/填充到 max_length → 转为 PyTorch tensor
        3. 将 tensor 移到目标设备（CPU/CUDA）
        4. 无梯度模式下执行前向推理，取 batch[0] 的 logits
        5. Softmax 归一化为概率分布，转回 CPU list
        6. 按概率从高到低排序，取最高分作为预测意图
        """

        # 第一步：格式化为"有/无历史对话。用户问题：xxx"的固定输入格式
        text = format_intent_model_input(query, has_history=has_history)
        # 第二步：tokenize → truncation + padding to max_length
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        # 第三步：将 input_ids/attention_mask 移到目标设备
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        # 第四步：torch.no_grad() 禁用梯度计算（推理模式，节省显存和计算）
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
            # 第五步：softmax 将 logits 转为概率分布，detach 后移至 CPU
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        # 第六步：构建 label → score 映射，取最高分作为预测意图
        scores = {label: float(probabilities[index]) for index, label in enumerate(self.labels)}
        intent = max(scores, key=scores.get)
        return IntentModelPrediction(
            intent=intent,
            score=scores[intent],
            scores=scores,
            reason="bert_intent_model",
            model_version=self.model_version,
        )

    def evaluate(self, examples: Iterable[IntentTrainingExample]) -> IntentModelEvaluation:
        """Evaluate the loaded BERT model on labeled examples.

        对标注样本逐条预测，统计准确率和混淆矩阵。
        混淆矩阵格式：matrix[真实标签][预测标签] = 计数
        """

        total = 0
        correct = 0
        # 初始化 3×3 混淆矩阵（FAQ_QUERY / KNOWLEDGE_QUERY / FOLLOW_UP）
        matrix: dict[str, dict[str, int]] = {
            label: {predicted: 0 for predicted in RETRIEVAL_INTENTS}
            for label in RETRIEVAL_INTENTS
        }
        for example in examples:
            if example.label not in matrix:
                raise ValueError(f"unsupported intent label: {example.label}")
            prediction = self.predict(example.query, has_history=example.has_history)
            total += 1
            if prediction.intent == example.label:
                correct += 1
            # 混淆矩阵：真实标签 → 预测标签 计数+1
            matrix[example.label][prediction.intent] += 1
        accuracy = correct / total if total else 0.0
        return IntentModelEvaluation(
            accuracy=accuracy,
            total=total,
            correct=correct,
            confusion_matrix=matrix,
        )


def _resolve_device(device: str) -> str:
    """解析设备参数：auto 时自动检测 CUDA 可用性，否则校验 cpu/cuda 合法性。"""
    normalized = (device or "cpu").strip().lower()
    # auto 模式：优先 CUDA，不可用时降级为 CPU
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("INTENT_MODEL_DEVICE=cuda，但当前环境不可用 CUDA")
    if normalized not in {"cpu", "cuda"}:
        raise RuntimeError("INTENT_MODEL_DEVICE 只能是 cpu、cuda 或 auto")
    return normalized


def _require_model_artifact(model_path: Path) -> None:
    """校验模型目录存在且包含必需的配置文件、标签文件和权重文件。"""
    if not model_path.exists():
        raise RuntimeError(f"意图 BERT 模型目录不存在：{model_path}")
    # 必需文件：HuggingFace config.json + 自定义 intent_labels.json
    required_files = ("config.json", LABELS_FILENAME)
    missing = [name for name in required_files if not (model_path / name).exists()]
    if missing:
        raise RuntimeError(f"意图 BERT 模型目录缺少文件 {missing}：{model_path}")
    # 权重文件兼容两种格式：safetensors（推荐）或 pytorch_model.bin（旧版）
    if not (model_path / "model.safetensors").exists() and not (model_path / "pytorch_model.bin").exists():
        raise RuntimeError(f"意图 BERT 模型目录缺少权重文件：{model_path}")


def _load_label_sequence(model_path: Path) -> tuple[str, ...]:
    """从 intent_labels.json 读取标签列表并校验顺序必须与 RETRIEVAL_INTENTS 一致。"""
    payload = json.loads((model_path / LABELS_FILENAME).read_text(encoding="utf-8"))
    # 兼容两种格式：{"labels": [...]} 或直接的 [...]
    if isinstance(payload, dict):
        raw_labels = payload.get("labels")
    else:
        raw_labels = payload
    if not isinstance(raw_labels, list) or not raw_labels:
        raise RuntimeError(f"{LABELS_FILENAME} 必须包含非空 labels 列表：{model_path}")
    labels = tuple(str(label).strip() for label in raw_labels if str(label).strip())
    # 标签顺序必须与训练时的 RETRIEVAL_INTENTS 一致，否则模型输出 index 对不上
    if labels != RETRIEVAL_INTENTS:
        raise RuntimeError(f"意图模型标签顺序必须是 {RETRIEVAL_INTENTS}，实际为 {labels}")
    return labels


def _read_model_version(model_path: Path) -> str:
    """从 intent_labels.json 读取模型版本号，未配置时返回默认版本。"""
    payload = json.loads((model_path / LABELS_FILENAME).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("model_version"):
        return str(payload["model_version"])
    return "bert-intent-v1"


DEFAULT_TRAINING_EXAMPLES: tuple[IntentTrainingExample, ...] = (
    IntentTrainingExample("新人入职需要哪些材料", "FAQ_QUERY"),
    IntentTrainingExample("新人入职流程有哪些", "FAQ_QUERY"),
    IntentTrainingExample("入职手续怎么办理", "FAQ_QUERY"),
    IntentTrainingExample("VPN 连不上怎么办", "FAQ_QUERY"),
    IntentTrainingExample("VPN 故障如何处理", "FAQ_QUERY"),
    IntentTrainingExample("发票什么时候可以开", "FAQ_QUERY"),
    IntentTrainingExample("发票开具需要什么资料", "FAQ_QUERY"),
    IntentTrainingExample("报销流程是什么", "FAQ_QUERY"),
    IntentTrainingExample("账号权限怎么申请", "FAQ_QUERY"),
    IntentTrainingExample("合同审批需要哪些材料", "FAQ_QUERY"),
    IntentTrainingExample("Webhook 调用失败怎么排查", "FAQ_QUERY"),
    IntentTrainingExample("如何重置密码", "FAQ_QUERY"),
    IntentTrainingExample("退费怎么申请", "FAQ_QUERY"),
    IntentTrainingExample("设备告警怎么处理", "FAQ_QUERY"),
    IntentTrainingExample("公司入职制度文档在哪里", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("请说明试用期转正规范", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("请解释员工离职权限回收制度", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("合同条款对付款有什么要求", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("隐私数据导出规范是什么", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("审计整改制度有哪些要求", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("帮我总结供应商尽调规范", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("API 限流规则说明", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("财务预算预审批制度", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("设备巡检规范", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("跨境制裁筛查规则", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("数据导出审批规范说明", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("那审批呢", "FOLLOW_UP", has_history=True),
    IntentTrainingExample("那要谁审批", "FOLLOW_UP", has_history=True),
    IntentTrainingExample("这个需要多久", "FOLLOW_UP", has_history=True),
    IntentTrainingExample("费用呢", "FOLLOW_UP", has_history=True),
    IntentTrainingExample("还有哪些材料", "FOLLOW_UP", has_history=True),
    IntentTrainingExample("上面这个怎么处理", "FOLLOW_UP", has_history=True),
    IntentTrainingExample("这个能不能延期", "FOLLOW_UP", has_history=True),
    IntentTrainingExample("那权限呢", "FOLLOW_UP", has_history=True),
    IntentTrainingExample("继续说明", "FOLLOW_UP", has_history=True),
    IntentTrainingExample("那个流程呢", "FOLLOW_UP", has_history=True),
    IntentTrainingExample("还有风险吗", "FOLLOW_UP", has_history=True),
)


DEFAULT_EVAL_EXAMPLES: tuple[IntentTrainingExample, ...] = (
    IntentTrainingExample("新人入职流程有哪些", "FAQ_QUERY"),
    IntentTrainingExample("VPN 故障如何处理", "FAQ_QUERY"),
    IntentTrainingExample("发票开具需要什么资料", "FAQ_QUERY"),
    IntentTrainingExample("账号权限回收怎么做", "FAQ_QUERY"),
    IntentTrainingExample("请解释员工离职权限回收制度", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("数据导出审批规范说明", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("总结一下合同付款条款要求", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("设备安全巡检制度", "KNOWLEDGE_QUERY"),
    IntentTrainingExample("那要谁审批", "FOLLOW_UP", has_history=True),
    IntentTrainingExample("这个材料还需要补充吗", "FOLLOW_UP", has_history=True),
    IntentTrainingExample("继续讲下一步", "FOLLOW_UP", has_history=True),
    IntentTrainingExample("费用怎么算", "FOLLOW_UP", has_history=True),
)
