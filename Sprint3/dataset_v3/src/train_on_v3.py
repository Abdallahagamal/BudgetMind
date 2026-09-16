"""Train the three classifiers on Complexity Dataset v3 (default: v3.3).

Uses the Sprint 2 pipeline unmodified — it imports Sprint2/src and only redirects
the embeddings directory to v3.3. Sprint 1 and Sprint 2 are not touched.

  python Sprint3/dataset_v3/src/train_on_v3.py --dimension all --save LogisticRegression
  python Sprint3/dataset_v3/src/train_on_v3.py --dimension type
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Sprint2" / "src"))
DEFAULT_VERSION = "3.3"

from data import DIMENSIONS, load_split, load_pooled          # noqa: E402
from train_classifier import run_dimension, candidates        # noqa: E402
import joblib, numpy as np                                     # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Train Type/Complexity/Domain on dataset v3.1.")
    p.add_argument("--dimension", choices=(*DIMENSIONS, "all"), default="all")
    p.add_argument("--save", metavar="MODEL_NAME",
                   help="Persist this candidate (e.g. LogisticRegression) per dimension.")
    p.add_argument("--skip-grouped", action="store_true")
    a = p.parse_args()
    tag = DEFAULT_VERSION.replace(".", "_")
    EMB = ROOT / "Sprint3" / "dataset_v3" / f"embeddings_v{tag}"
    OUT_RESULTS = ROOT / "Sprint3" / "dataset_v3" / f"results_v{tag}"
    OUT_MODELS = ROOT / "Sprint3" / "dataset_v3" / f"models_v{tag}"

    if not EMB.exists():
        print(f"ERROR: embeddings not found at {EMB}")
        return 1
    train, validation, test = (load_split(s, embeddings_dir=EMB)
                               for s in ("train", "validation", "test"))
    pooled = load_pooled(embeddings_dir=EMB)
    print(f"dataset v{DEFAULT_VERSION} · train={len(train)} validation={len(validation)} test={len(test)} "
          f"pooled={len(pooled)} sources={len(set(pooled.groups))}")

    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    dims = DIMENSIONS if a.dimension == "all" else (a.dimension,)
    for dim in dims:
        res = run_dimension(dim, train, validation, test, pooled,
                            skip_grouped=a.skip_grouped)
        (OUT_RESULTS / f"{dim}_results.json").write_text(json.dumps(res, indent=2),
                                                         encoding="utf-8")
        print(f"  results -> {OUT_RESULTS.relative_to(ROOT)}/{dim}_results.json")
        if a.save:
            if a.save not in candidates():
                print(f"  ERROR: unknown candidate {a.save!r}"); return 1
            model = candidates()[a.save]
            X = np.vstack([train.X, validation.X])
            y = np.concatenate([train.y(dim), validation.y(dim)])
            model.fit(X, y)
            OUT_MODELS.mkdir(parents=True, exist_ok=True)
            path = OUT_MODELS / f"{dim}_classifier_v{tag}.pkl"
            joblib.dump(model, path)
            print(f"  model   -> {OUT_MODELS.relative_to(ROOT)}/{path.name} "
                  f"({a.save}, fit on train+validation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
