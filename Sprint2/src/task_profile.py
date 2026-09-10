from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np

TAXONOMY_VERSION = "1.0"


class ProbabilisticClassifier(Protocol):
    classes_: np.ndarray

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    margin: float


def predict_with_margin(model: ProbabilisticClassifier, embedding: np.ndarray) -> Prediction:
    X = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
    proba = model.predict_proba(X)[0]
    order = np.argsort(proba)[::-1]

    top = float(proba[order[0]])
    runner_up = float(proba[order[1]]) if len(order) > 1 else 0.0

    return Prediction(
        label=str(model.classes_[order[0]]),
        confidence=top,
        margin=top - runner_up,
    )


@dataclass(frozen=True)
class TaskProfile:
    task_id: str
    taxonomy_version: str

    type: str
    complexity: str
    domain: str

    type_confidence: float
    complexity_confidence: float
    domain_confidence: float

    type_margin: float
    complexity_margin: float
    domain_margin: float

    embedding_ref: str
    classified_at: str
    from_cache: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def context_bucket(self) -> tuple[str, str, str]:
        return (self.type, self.complexity, self.domain)

    def min_margin(self) -> float:
        return min(self.type_margin, self.complexity_margin, self.domain_margin)


def build_task_profile(
    task_id: str,
    embedding: np.ndarray,
    *,
    type_model: ProbabilisticClassifier,
    complexity_model: ProbabilisticClassifier,
    domain_model: ProbabilisticClassifier,
    from_cache: bool = False,
    embedding_ref: str | None = None,
    taxonomy_version: str = TAXONOMY_VERSION,
) -> TaskProfile:
    type_pred = predict_with_margin(type_model, embedding)
    complexity_pred = predict_with_margin(complexity_model, embedding)
    domain_pred = predict_with_margin(domain_model, embedding)

    return TaskProfile(
        task_id=task_id,
        taxonomy_version=taxonomy_version,
        type=type_pred.label,
        complexity=complexity_pred.label,
        domain=domain_pred.label,
        type_confidence=round(type_pred.confidence, 4),
        complexity_confidence=round(complexity_pred.confidence, 4),
        domain_confidence=round(domain_pred.confidence, 4),
        type_margin=round(type_pred.margin, 4),
        complexity_margin=round(complexity_pred.margin, 4),
        domain_margin=round(domain_pred.margin, 4),
        embedding_ref=embedding_ref or f"emb:{task_id}",
        classified_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        from_cache=from_cache,
    )
