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
from evaluation import evaluate_predictions, format_evaluation  # noqa: E402
from task_profile import predict_with_margin  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DIMENSION = "domain"
TOLERANCE = 0.01  # accuracy/F1 points; guards against silent drift, not noise


def check(label: str, ok: bool, detail: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label} — {detail}")
    return ok


def main() -> int:
    print(f"{'=' * 78}\n  DOMAIN — Sprint 3 accuracy test (regression check vs Sprint 2)\n{'=' * 78}\n")
    all_ok = True

    test = load_split("test")
    y_test = test.y(DIMENSION)

    # --- 1. Data integrity ---
    print("Data integrity")
    norms = np.linalg.norm(test.X, axis=1)
    all_ok &= check("embedding dimension", test.X.shape[1] == 384, f"got {test.X.shape[1]}")
    all_ok &= check("L2 normalization", np.allclose(norms, 1.0, atol=1e-3),
                     f"norm range [{norms.min():.4f}, {norms.max():.4f}]")
    all_ok &= check("test set size", len(test) == 83, f"got {len(test)}")

    sprint2_results = json.loads(
        (REPO_ROOT / "Sprint2" / "results" / "domain_results.json").read_text(encoding="utf-8")
    )
    recorded = sprint2_results["candidates"]["LogisticRegression"]["random_protocol"]

    # --- 2. Sprint 2 baseline reproduces its recorded numbers ---
    print("\nSprint 2 baseline reproduction")
    baseline = joblib.load(REPO_ROOT / "Sprint2" / "models" / "domain_classifier_baseline.pkl")
    proba = baseline.predict_proba(test.X)
    pred = np.asarray([baseline.classes_[i] for i in proba.argmax(axis=1)])
    ev = evaluate_predictions(y_test, pred, model_name="baseline", dimension=DIMENSION,
                               labels=sorted(set(y_test) | set(pred)))

    all_ok &= check(
        "test accuracy matches recorded",
        abs(ev.accuracy - recorded["test_accuracy"]) <= TOLERANCE,
        f"live={ev.accuracy:.4f}  recorded={recorded['test_accuracy']:.4f}",
    )
    all_ok &= check(
        "test macro-F1 matches recorded",
        abs(ev.macro_f1 - recorded["test_macro_f1"]) <= TOLERANCE,
        f"live={ev.macro_f1:.4f}  recorded={recorded['test_macro_f1']:.4f}",
    )

    # --- 3. Interface contract still holds ---
    print("\nInterface contract")
    sample = predict_with_margin(baseline, test.X[0])
    all_ok &= check(
        "predict_with_margin returns valid fields",
        isinstance(sample.label, str) and 0.0 <= sample.confidence <= 1.0 and 0.0 <= sample.margin <= 1.0,
        f"label={sample.label!r} confidence={sample.confidence:.4f} margin={sample.margin:.4f}",
    )

    refined_path = Path(__file__).resolve().parent.parent / "models" / "domain_classifier_refined.pkl"
    report = {
        "dimension": DIMENSION,
        "all_checks_passed": all_ok,
        "baseline": {"live_accuracy": ev.accuracy, "live_macro_f1": ev.macro_f1,
                     "recorded_accuracy": recorded["test_accuracy"],
                     "recorded_macro_f1": recorded["test_macro_f1"]},
    }

    if refined_path.exists():
        print("\nSprint 3 refined model (informational, not a regression check vs Sprint 2)")
        refined = joblib.load(refined_path)
        r_proba = refined.predict_proba(test.X)
        r_pred = np.asarray([refined.classes_[i] for i in r_proba.argmax(axis=1)])
        r_ev = evaluate_predictions(y_test, r_pred, model_name="refined", dimension=DIMENSION,
                                     labels=sorted(set(y_test) | set(r_pred)))
        print(format_evaluation(r_ev))
        report["refined"] = {"live_accuracy": r_ev.accuracy, "live_macro_f1": r_ev.macro_f1}

    print(f"\n{'=' * 78}\n  Overall: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}\n{'=' * 78}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "domain_accuracy_test.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved to {out.relative_to(REPO_ROOT)}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
