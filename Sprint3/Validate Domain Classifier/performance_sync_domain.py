from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SPRINT2_SRC = REPO_ROOT / "Sprint2" / "src"
sys.path.insert(0, str(SPRINT2_SRC))

from data import load_split  # noqa: E402
from evaluation import evaluate_predictions, expected_calibration_error  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DIMENSION = "domain"


def summarize(model_path: Path, model_label: str, X_test, y_test, labels) -> dict:
    model = joblib.load(model_path)
    proba = model.predict_proba(X_test)
    pred = np.asarray([model.classes_[i] for i in proba.argmax(axis=1)])
    ev = evaluate_predictions(y_test, pred, model_name=model_label, dimension=DIMENSION, labels=labels)
    ece = expected_calibration_error(y_test, proba, list(model.classes_))
    return {
        "model": model_label,
        "accuracy": round(ev.accuracy, 4),
        "macro_precision": round(ev.macro_precision, 4),
        "macro_recall": round(ev.macro_recall, 4),
        "macro_f1": round(ev.macro_f1, 4),
        "ece": round(ece, 4),
        "n_test": len(y_test),
    }


def main() -> int:
    test = load_split("test")
    y_test = test.y(DIMENSION)
    labels = sorted(set(y_test))

    sync = {"dimension": DIMENSION, "owner": "Nancy", "sprint": 3, "entries": []}

    sync["entries"].append(summarize(
        REPO_ROOT / "Sprint2" / "models" / "domain_classifier_baseline.pkl",
        "Sprint2_baseline (LogisticRegression, uncalibrated)", test.X, y_test, labels,
    ))

    refined_path = Path(__file__).resolve().parent.parent / "models" / "domain_classifier_refined.pkl"
    if refined_path.exists():
        sync["entries"].append(summarize(
            refined_path, "Sprint3_refined (LogisticRegression + sigmoid calibration)",
            test.X, y_test, labels,
        ))

    print(f"DOMAIN — performance sync ({len(sync['entries'])} entries)\n")
    for e in sync["entries"]:
        print(f"  {e['model']:<48} acc={e['accuracy']:.3f}  macroF1={e['macro_f1']:.3f}  ECE={e['ece']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "domain_performance_sync.json"
    out.write_text(json.dumps(sync, indent=2), encoding="utf-8")
    print(f"\nSaved to {out.relative_to(REPO_ROOT)} — ready to merge with M1/M2's equivalents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
