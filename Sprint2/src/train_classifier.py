from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import DIMENSIONS, EMBEDDING_MODEL, Dataset, load_pooled, load_split
from evaluation import (
    error_rows,
    evaluate_predictions,
    expected_calibration_error,
    format_evaluation,
    grouped_evaluation,
    majority_baseline,
)
from task_profile import TAXONOMY_VERSION  

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "Sprint2" / "models"
DEFAULT_RESULTS_DIR = REPO_ROOT / "Sprint2" / "results"

RANDOM_SEED = 42


def candidates() -> dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=3000, C=1.0, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        ),
        "SVM": SVC(
            C=10.0, kernel="rbf", probability=True,
            class_weight="balanced", random_state=RANDOM_SEED,
        ),
    }


def run_dimension(
    dimension: str,
    train: Dataset,
    validation: Dataset,
    test: Dataset,
    pooled: Dataset,
    *,
    skip_grouped: bool = False,
) -> dict[str, Any]:
    y_train, y_val, y_test = train.y(dimension), validation.y(dimension), test.y(dimension)
    labels = sorted(set(pooled.y(dimension)))

    maj_acc, maj_f1 = majority_baseline(y_train, y_test)

    print(f"\n{'=' * 74}")
    print(f"  {dimension.upper()}   n_train={len(train)}  n_val={len(validation)}  n_test={len(test)}")
    print(f"  classes: {', '.join(labels)}")
    print(f"{'=' * 74}")
    print(f"\n  Majority-class baseline   acc={maj_acc:.3f}  macroF1={maj_f1:.3f}")

    results: dict[str, Any] = {
        "dimension": dimension,
        "taxonomy_version": TAXONOMY_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "labels": labels,
        "majority_baseline": {"accuracy": maj_acc, "macro_f1": maj_f1},
        "candidates": {},
    }

    print("\n  --- PROTOCOL 1: RANDOM (Sprint 1 stratified split) ---")
    print("  Optimistic: source predicts label almost perfectly, so a model can")
    print("  score high here by recognising dataset style. Not the selection metric.\n")

    fitted: dict[str, Any] = {}
    for name, estimator in candidates().items():
        estimator.fit(train.X, y_train)
        fitted[name] = estimator

        start = time.perf_counter()
        proba = estimator.predict_proba(test.X)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        pred = np.asarray([estimator.classes_[i] for i in proba.argmax(axis=1)])

        ev = evaluate_predictions(
            y_test, pred, model_name=name, dimension=dimension, labels=labels
        )
        ev.majority_accuracy, ev.majority_macro_f1 = maj_acc, maj_f1
        ev.inference_ms = elapsed_ms
        ev.ece = expected_calibration_error(y_test, proba, list(estimator.classes_))

        val_proba = estimator.predict_proba(validation.X)
        val_pred = np.asarray([estimator.classes_[i] for i in val_proba.argmax(axis=1)])
        val_ev = evaluate_predictions(
            y_val, val_pred, model_name=name, dimension=dimension, labels=labels
        )

        print(format_evaluation(ev))
        print(f"    validation macroF1={val_ev.macro_f1:.3f}\n")

        results["candidates"][name] = {
            "random_protocol": {
                "validation_macro_f1": val_ev.macro_f1,
                "test_accuracy": ev.accuracy,
                "test_macro_precision": ev.macro_precision,
                "test_macro_recall": ev.macro_recall,
                "test_macro_f1": ev.macro_f1,
                "test_ece": ev.ece,
                "inference_ms_for_test_set": elapsed_ms,
                "per_class": [vars(c) for c in ev.per_class],
                "confusion_matrix": ev.confusion.tolist(),
            }
        }

        errors = error_rows(
            test.task_ids, y_test, pred, proba, list(estimator.classes_)
        )
        results["candidates"][name]["random_protocol"]["errors"] = errors

    if skip_grouped:
        return results

    print("  --- PROTOCOL 2: GROUPED (leave-one-source-out) ---")
    print("  The selection metric. Trains without one benchmark source and scores")
    print("  it, so dataset style cannot be used as a shortcut.\n")

    for name, estimator in candidates().items():
        ev, per_source = grouped_evaluation(
            pooled.X,
            pooled.y(dimension),
            pooled.groups,
            estimator,
            model_name=name,
            dimension=dimension,
        )
        print(format_evaluation(ev))
        if ev.extras["unevaluable_labels"]:
            print(f"    not evaluable under this protocol (single-source labels): "
                  f"{', '.join(ev.extras['unevaluable_labels'])}")
        print(f"    {'held-out source':<44}{'n':>5}{'acc':>8}")
        for row in sorted(per_source, key=lambda r: r["accuracy"]):
            flag = "" if row["label_covered_by_train"] else "  (label unseen in train)"
            print(f"    {row['source']:<44}{row['n']:>5}{row['accuracy']:>8.3f}{flag}")
        print()

        results["candidates"][name]["grouped_protocol"] = {
            "pooled_accuracy": ev.accuracy,
            "pooled_macro_f1": ev.macro_f1,
            "evaluated_labels": ev.extras["evaluated_labels"],
            "unevaluable_labels": ev.extras["unevaluable_labels"],
            "per_class": [vars(c) for c in ev.per_class],
            "per_source": per_source,
        }

    print("  --- GENERALIZATION GAP (report both numbers in the thesis) ---")
    print(f"    {'model':<22}{'random F1':>12}{'grouped F1':>13}{'gap':>9}")
    for name, payload in results["candidates"].items():
        rnd = payload["random_protocol"]["test_macro_f1"]
        grp = payload["grouped_protocol"]["pooled_macro_f1"]
        print(f"    {name:<22}{rnd:>12.3f}{grp:>13.3f}{rnd - grp:>9.3f}")

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and compare Sprint 2 classifier candidates.")
    p.add_argument("--dimension", choices=(*DIMENSIONS, "all"), default="all")
    p.add_argument("--save", metavar="MODEL_NAME",
                   help="Persist this candidate as the finalized baseline (e.g. LogisticRegression).")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--skip-grouped", action="store_true",
                   help="Skip leave-one-source-out (faster, but not a reportable result).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    train, validation, test = load_split("train"), load_split("validation"), load_split("test")
    pooled = load_pooled()

    dimensions = DIMENSIONS if args.dimension == "all" else (args.dimension,)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    for dimension in dimensions:
        results = run_dimension(
            dimension, train, validation, test, pooled, skip_grouped=args.skip_grouped
        )
        out = args.results_dir / f"{dimension}_results.json"
        if args.skip_grouped and out.exists():
            previous = json.loads(out.read_text(encoding="utf-8"))
            for name, payload in previous.get("candidates", {}).items():
                if "grouped_protocol" in payload and name in results["candidates"]:
                    results["candidates"][name]["grouped_protocol"] = payload["grouped_protocol"]
            print("\n  (--skip-grouped: preserved existing grouped_protocol results)")
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  results -> {out.relative_to(REPO_ROOT)}")

        if args.save:
            if args.save not in candidates():
                print(f"  ERROR: unknown candidate {args.save!r}")
                return 1
            model = candidates()[args.save]
            
            X = np.vstack([train.X, validation.X])
            y = np.concatenate([train.y(dimension), validation.y(dimension)])
            model.fit(X, y)

            args.model_dir.mkdir(parents=True, exist_ok=True)
            path = args.model_dir / f"{dimension}_classifier_baseline.pkl"
            joblib.dump(model, path)
            print(f"  model   -> {path.relative_to(REPO_ROOT)}  ({args.save}, fit on train+validation)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
