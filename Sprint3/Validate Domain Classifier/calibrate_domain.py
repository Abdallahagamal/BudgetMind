from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# Windows consoles often default to a codepage (e.g. cp1252) that can't
# encode the Δ character used in the summary print below; force UTF-8 for
# stdout so the script runs the same way on every OS.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
SPRINT2_SRC = REPO_ROOT / "Sprint2" / "src"
sys.path.insert(0, str(SPRINT2_SRC))

from data import load_split, load_pooled  # noqa: E402
from evaluation import (  # noqa: E402
    evaluate_predictions,
    expected_calibration_error,
    format_evaluation,
)
from task_profile import predict_with_margin  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DIMENSION = "domain"
RANDOM_SEED = 42

CALIBRATION_GRID = [
    {"method": "sigmoid", "cv": 3},
    {"method": "sigmoid", "cv": 5},
    {"method": "isotonic", "cv": 3},
    {"method": "isotonic", "cv": 5},
]

# Isotonic regression is non-parametric and known to overfit when the
# per-class calibration sample is small (sklearn's own docs recommend Platt/
# sigmoid below roughly 1,000 examples per class). DOMAIN's smallest class
# (Mathematics & Quantitative) has only 40 examples total, ~34 available for
# calibration after holding out test (28 train + 6 validation) — even
# smaller than TYPE's Summarization (29 total). So sigmoid is the a priori
# preferred family here — decided before looking at test, not after.
# Isotonic is still tried and reported for comparison, since the risk is
# worth demonstrating rather than just asserting.
PREFERRED_FAMILY = "sigmoid"


def base_estimator() -> LogisticRegression:
    return LogisticRegression(
        max_iter=3000, C=1.0, class_weight="balanced", random_state=RANDOM_SEED
    )


def margin_gate_table(model, X_test, y_test, labels, thresholds=(0.0, 0.3, 0.5)):
    """Reproduce the README's margin-gate table for this model's margins."""
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    rows = []
    for i in range(len(X_test)):
        p = proba[i]
        order = np.argsort(p)[::-1]
        top, runner_up = float(p[order[0]]), float(p[order[1]])
        pred = classes[order[0]]
        rows.append({"margin": top - runner_up, "correct": pred == y_test[i]})

    table = []
    for t in thresholds:
        admitted = [r for r in rows if r["margin"] >= t]
        acc = float(np.mean([r["correct"] for r in admitted])) if admitted else float("nan")
        table.append({
            "min_margin": t,
            "admitted": len(admitted),
            "admitted_pct": len(admitted) / len(rows),
            "accuracy_among_admitted": acc,
        })
    return table


def main() -> int:
    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")
    pooled = load_pooled()

    y_train, y_val, y_test = train.y(DIMENSION), validation.y(DIMENSION), test.y(DIMENSION)
    labels = sorted(set(pooled.y(DIMENSION)))

    X_trainval = np.vstack([train.X, validation.X])
    y_trainval = np.concatenate([y_train, y_val])

    print(f"{'=' * 78}\n  DOMAIN — Sprint 3 model refinement: probability calibration\n{'=' * 78}")

    # ---- Baseline: Sprint 2's committed model, loaded (not refit) ----
    baseline_path = REPO_ROOT / "Sprint2" / "models" / "domain_classifier_baseline.pkl"
    baseline = joblib.load(baseline_path)
    print(f"\nLoaded Sprint 2 baseline from {baseline_path.relative_to(REPO_ROOT)}")

    # ---- Calibration search — selection happens WITHOUT touching test ----
    # CalibratedClassifierCV internally cross-validates on X_trainval, so this
    # is honest: no calibration setting is chosen by looking at test.
    print("\n--- Calibration search (internal CV on train+validation only) ---")
    candidates = {}
    for cfg in CALIBRATION_GRID:
        name = f"{cfg['method']}_cv{cfg['cv']}"
        model = CalibratedClassifierCV(base_estimator(), method=cfg["method"], cv=cfg["cv"])
        t0 = time.perf_counter()
        model.fit(X_trainval, y_trainval)
        fit_s = time.perf_counter() - t0

        # Internal check: held-out-style ECE using the val split alone (the
        # calibrator was fit on train+val jointly via internal CV, so this is
        # a soft check, not a clean held-out number — the real number is the
        # one-time test check below).
        val_proba = model.predict_proba(validation.X)
        val_pred = np.asarray([model.classes_[i] for i in val_proba.argmax(axis=1)])
        val_ev = evaluate_predictions(y_val, val_pred, model_name=name, dimension=DIMENSION, labels=labels)
        val_ece = expected_calibration_error(y_val, val_proba, list(model.classes_))

        print(f"  {name:<18} fit={fit_s:.2f}s  val_macroF1={val_ev.macro_f1:.3f}  val_ECE={val_ece:.3f}")
        candidates[name] = {"model": model, "val_macro_f1": val_ev.macro_f1, "val_ece": val_ece}

    # Select: lowest val ECE among candidates that don't regress val Macro-F1
    # by more than 0.02 versus the uncalibrated baseline refit on the same data.
    baseline_trainval = base_estimator().fit(X_trainval, y_trainval)
    baseline_val_proba = baseline_trainval.predict_proba(validation.X)
    baseline_val_pred = np.asarray(
        [baseline_trainval.classes_[i] for i in baseline_val_proba.argmax(axis=1)]
    )
    baseline_val_ev = evaluate_predictions(
        y_val, baseline_val_pred, model_name="uncalibrated", dimension=DIMENSION, labels=labels
    )
    baseline_val_ece = expected_calibration_error(
        y_val, baseline_val_proba, list(baseline_trainval.classes_)
    )
    print(f"\n  {'uncalibrated (refit)':<18} val_macroF1={baseline_val_ev.macro_f1:.3f}  val_ECE={baseline_val_ece:.3f}")

    f1_floor = baseline_val_ev.macro_f1 - 0.02
    eligible = {k: v for k, v in candidates.items() if v["val_macro_f1"] >= f1_floor}
    if not eligible:
        eligible = candidates  # guardrail: never end up with nothing to pick

    # Restrict to the a priori preferred family (sigmoid, chosen for the
    # small-sample reason above) and pick the lowest val ECE within it.
    preferred = {k: v for k, v in eligible.items() if k.startswith(PREFERRED_FAMILY)}
    pool = preferred if preferred else eligible
    winner_name = min(pool, key=lambda k: pool[k]["val_ece"])
    winner = candidates[winner_name]["model"]
    print(f"\n>>> Selected calibration: {winner_name} (val_ECE={candidates[winner_name]['val_ece']:.3f}, "
          f"val_macroF1={candidates[winner_name]['val_macro_f1']:.3f})")
    print(f"    Restricted to '{PREFERRED_FAMILY}' family a priori — see PREFERRED_FAMILY comment above.")

    # ---- Touch TEST exactly once: baseline + every calibration candidate,
    # reported together. This is a single one-time evaluation for transparency
    # (including the non-selected candidates as a documented comparison), not
    # an iterative re-selection loop — the winner above was already fixed
    # using validation only, before this block runs. ----
    print("\n--- Final TEST comparison (touched once, all candidates reported) ---")
    report = {"dimension": DIMENSION, "selected_calibration": winner_name,
              "selection_rationale": (
                  f"Restricted a priori to '{PREFERRED_FAMILY}' calibration "
                  "(small per-class sample, esp. Mathematics & Quantitative "
                  "n=40, makes isotonic prone to overfitting), then picked "
                  "lowest validation ECE within that family. Does not address "
                  "Domain's data-coverage limitations (Mathematics & "
                  "Quantitative, Language & Communication) — those require "
                  "new data, not calibration."
              ), "models": {}}

    all_models = {"baseline_uncalibrated": baseline, **{k: v["model"] for k, v in candidates.items()}}
    for name, model in all_models.items():
        t0 = time.perf_counter()
        proba = model.predict_proba(test.X)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        pred = np.asarray([model.classes_[i] for i in proba.argmax(axis=1)])
        ev = evaluate_predictions(y_test, pred, model_name=name, dimension=DIMENSION, labels=labels)
        ece = expected_calibration_error(y_test, proba, list(model.classes_))
        gate = margin_gate_table(model, test.X, y_test, labels)

        print(f"\n  {name}")
        print(format_evaluation(ev))
        print(f"    test_ECE={ece:.3f}  inference={infer_ms:.2f}ms")
        print(f"    margin gate: " + "  ".join(
            f"t={g['min_margin']}: {g['admitted']}/{len(test.X)} admitted, "
            f"acc={g['accuracy_among_admitted']:.3f}" for g in gate
        ))

        report["models"][name] = {
            "test_accuracy": ev.accuracy,
            "test_macro_precision": ev.macro_precision,
            "test_macro_recall": ev.macro_recall,
            "test_macro_f1": ev.macro_f1,
            "test_ece": ece,
            "inference_ms_for_test_set": infer_ms,
            "margin_gate": gate,
            "per_class": [vars(c) for c in ev.per_class],
        }

    delta_f1 = (report["models"][winner_name]["test_macro_f1"]
                - report["models"]["baseline_uncalibrated"]["test_macro_f1"])
    delta_ece = (report["models"][winner_name]["test_ece"]
                 - report["models"]["baseline_uncalibrated"]["test_ece"])
    report["delta_test_macro_f1"] = delta_f1
    report["delta_test_ece"] = delta_ece

    print(f"\n{'=' * 78}\n  Selected ({winner_name}) vs baseline on TEST: "
          f"ΔMacro-F1={delta_f1:+.3f}  ΔECE={delta_ece:+.3f}\n{'=' * 78}")

    # ---- Save refined model + report (does NOT overwrite Sprint 2 baseline) ----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    refined_path = MODELS_DIR / "domain_classifier_refined.pkl"
    joblib.dump(winner, refined_path)
    print(f"\nRefined model saved to {refined_path.relative_to(REPO_ROOT)} "
          f"(Sprint 2 baseline left untouched — swap is a team decision).")

    # Contract check: predict_with_margin still works on the refined model
    sample_pred = predict_with_margin(winner, test.X[0])
    print(f"Contract check: {sample_pred}")
    report["contract_check"] = {
        "label": sample_pred.label,
        "confidence": round(sample_pred.confidence, 4),
        "margin": round(sample_pred.margin, 4),
    }

    out = RESULTS_DIR / "domain_refinement_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved to {out.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
