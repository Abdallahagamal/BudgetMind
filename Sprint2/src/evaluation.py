from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class ClassReport:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class Evaluation:
    model_name: str
    dimension: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: list[ClassReport]
    labels: list[str]
    confusion: np.ndarray
    ece: float | None = None
    majority_accuracy: float | None = None
    majority_macro_f1: float | None = None
    inference_ms: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def summary_line(self) -> str:
        lead = f"{self.model_name:<22} acc={self.accuracy:.3f}  macroF1={self.macro_f1:.3f}"
        if self.majority_macro_f1 is not None:
            lead += f"  (majority macroF1={self.majority_macro_f1:.3f})"
        if self.ece is not None:
            lead += f"  ECE={self.ece:.3f}"
        if self.inference_ms is not None:
            lead += f"  {self.inference_ms:.2f}ms"
        return lead


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    model_name: str,
    dimension: str,
    labels: Sequence[str] | None = None,
) -> Evaluation:
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    return Evaluation(
        model_name=model_name,
        dimension=dimension,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        per_class=[
            ClassReport(labels[i], float(precision[i]), float(recall[i]), float(f1[i]), int(support[i]))
            for i in range(len(labels))
        ],
        labels=labels,
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
    )


def majority_baseline(
    y_train: Sequence[str], y_test: Sequence[str]
) -> tuple[float, float]:
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    pred = dummy.predict(np.zeros((len(y_test), 1)))
    return (
        float(accuracy_score(y_test, pred)),
        float(f1_score(y_test, pred, average="macro", zero_division=0)),
    )


def expected_calibration_error(
    y_true: Sequence[str],
    proba: np.ndarray,
    classes: Sequence[str],
    n_bins: int = 10,
) -> float:
    classes = list(classes)
    y_true = np.asarray(y_true)
    confidence = proba.max(axis=1)
    predicted = np.asarray([classes[i] for i in proba.argmax(axis=1)])
    correct = (predicted == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def grouped_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator: Any,
    *,
    model_name: str,
    dimension: str,
    min_group_size: int = 5,
) -> tuple[Evaluation, list[dict[str, Any]]]:
    labels = sorted(set(y))
    pooled_true: list[str] = []
    pooled_pred: list[str] = []
    per_source: list[dict[str, Any]] = []

    for source in sorted(set(groups)):
        held = groups == source
        if held.sum() < min_group_size:
            continue
        train_labels = set(y[~held])
        coverable = set(y[held]) <= train_labels

        model = clone(estimator)
        model.fit(X[~held], y[~held])
        pred = model.predict(X[held])

        per_source.append(
            {
                "source": str(source),
                "n": int(held.sum()),
                "true_labels": sorted(set(y[held])),
                "accuracy": float(accuracy_score(y[held], pred)),
                "label_covered_by_train": bool(coverable),
                "top_confusions": _top_confusions(y[held], pred),
            }
        )
        if coverable:
            pooled_true.extend(y[held].tolist())
            pooled_pred.extend(pred.tolist())

    evaluated_labels = [l for l in labels if l in set(pooled_true)]
    unevaluable = [l for l in labels if l not in evaluated_labels]

    evaluation = evaluate_predictions(
        pooled_true,
        pooled_pred,
        model_name=model_name,
        dimension=dimension,
        labels=evaluated_labels,
    )
    evaluation.extras["protocol"] = "grouped_leave_one_source_out"
    evaluation.extras["per_source"] = per_source
    evaluation.extras["evaluated_labels"] = evaluated_labels
    evaluation.extras["unevaluable_labels"] = unevaluable
    return evaluation, per_source


def _top_confusions(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> list[dict[str, Any]]:
    wrong = y_pred[y_pred != y_true]
    values, counts = np.unique(wrong, return_counts=True)
    order = np.argsort(-counts)[:k]
    return [{"predicted": str(values[i]), "count": int(counts[i])} for i in order]


def error_rows(
    task_ids: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    proba: np.ndarray | None,
    classes: Sequence[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true == pred:
            continue
        row: dict[str, Any] = {
            "task_id": str(task_ids[i]),
            "true": str(true),
            "predicted": str(pred),
        }
        if proba is not None and classes is not None:
            ordered = np.sort(proba[i])[::-1]
            row["confidence"] = float(ordered[0])
            row["margin"] = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0
        rows.append(row)
    return rows


def format_evaluation(ev: Evaluation) -> str:
    out: list[str] = [ev.summary_line()]
    out.append(f"    macro precision={ev.macro_precision:.3f}  macro recall={ev.macro_recall:.3f}")
    out.append(f"    {'class':<34}{'prec':>7}{'rec':>7}{'f1':>7}{'support':>9}")
    for c in ev.per_class:
        out.append(
            f"    {c.label:<34}{c.precision:>7.3f}{c.recall:>7.3f}{c.f1:>7.3f}{c.support:>9d}"
        )
    return "\n".join(out)
