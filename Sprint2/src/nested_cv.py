from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import DIMENSIONS, load_pooled
from evaluation import evaluate_predictions

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "Sprint2" / "results"

RANDOM_SEED = 42
C_GRID = (0.1, 1.0, 10.0, 100.0)
MIN_GROUP_SIZE = 5


def _make(C: float) -> LogisticRegression:
    return LogisticRegression(
        max_iter=3000, C=C, class_weight="balanced", random_state=RANDOM_SEED
    )


def _select_C(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 3) -> float:
    n_groups = len(set(groups))
    if n_groups < 2:
        return 1.0
    splitter = GroupKFold(n_splits=min(n_splits, n_groups))

    scores: dict[float, list[float]] = {C: [] for C in C_GRID}
    for inner_train, inner_val in splitter.split(X, y, groups):
        for C in C_GRID:
            model = _make(C).fit(X[inner_train], y[inner_train])
            pred = model.predict(X[inner_val])
            scores[C].append(f1_score(y[inner_val], pred, average="macro", zero_division=0))

    return max(C_GRID, key=lambda C: float(np.mean(scores[C])))


def nested_grouped_cv(dimension: str) -> dict[str, Any]:
    pooled = load_pooled()
    X, y, groups = pooled.X, pooled.y(dimension), pooled.groups
    labels = sorted(set(y))

    pooled_true: list[str] = []
    pooled_pred: list[str] = []
    chosen: list[dict[str, Any]] = []

    for source in sorted(set(groups)):
        held = groups == source
        if held.sum() < MIN_GROUP_SIZE:
            continue
        if not set(y[held]) <= set(y[~held]):
            continue

        C = _select_C(X[~held], y[~held], groups[~held])
        model = _make(C).fit(X[~held], y[~held])
        pred = model.predict(X[held])

        pooled_true.extend(y[held].tolist())
        pooled_pred.extend(pred.tolist())
        chosen.append(
            {
                "held_out_source": str(source),
                "n": int(held.sum()),
                "selected_C": C,
                "accuracy": float((pred == y[held]).mean()),
            }
        )

    evaluated = [l for l in labels if l in set(pooled_true)]
    ev = evaluate_predictions(
        pooled_true, pooled_pred,
        model_name="LogisticRegression (nested CV)",
        dimension=dimension, labels=evaluated,
    )

    counts = Counter(c["selected_C"] for c in chosen)
    return {
        "dimension": dimension,
        "protocol": "nested_leave_one_source_out",
        "C_grid": list(C_GRID),
        "nested_macro_f1": ev.macro_f1,
        "nested_accuracy": ev.accuracy,
        "evaluated_labels": evaluated,
        "unevaluable_labels": [l for l in labels if l not in evaluated],
        "selected_C_counts": {str(k): v for k, v in sorted(counts.items())},
        "modal_C": float(counts.most_common(1)[0][0]),
        "per_fold": chosen,
        "per_class": [vars(c) for c in ev.per_class],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested grouped CV for honest hyperparameter selection.")
    p.add_argument("--dimension", choices=(*DIMENSIONS, "all"), default="all")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dimensions = DIMENSIONS if args.dimension == "all" else (args.dimension,)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    for dimension in dimensions:
        r = nested_grouped_cv(dimension)
        print(f"\n{'=' * 72}\n  {dimension.upper()} — nested leave-one-source-out\n{'=' * 72}")
        print(f"  nested macro-F1 = {r['nested_macro_f1']:.3f}   accuracy = {r['nested_accuracy']:.3f}")
        print(f"  C chosen per fold: {r['selected_C_counts']}   modal C = {r['modal_C']}")
        if r["unevaluable_labels"]:
            print(f"  not evaluable (single-source labels): {', '.join(r['unevaluable_labels'])}")
        print(f"\n  {'held-out source':<44}{'n':>5}{'C':>8}{'acc':>8}")
        for f in sorted(r["per_fold"], key=lambda x: x["accuracy"]):
            print(f"  {f['held_out_source']:<44}{f['n']:>5}{f['selected_C']:>8}{f['accuracy']:>8.3f}")

        out = args.results_dir / f"{dimension}_nested_cv.json"
        out.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"\n  -> {out.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
