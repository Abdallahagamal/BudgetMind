"""Final evaluation of all three dimensions on the canonical dataset.

  python Sprint3/dataset_v3/src/final_evaluation.py
"""
from __future__ import annotations
import json, collections, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "Sprint3" / "dataset_v3"
VERSION = "3.3"
TAG = VERSION.replace(".", "_")
LR = lambda: LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=42)


def main() -> int:
    rows = [json.loads(l) for l in (D / f"complexity_dataset_v{TAG}.jsonl").open(encoding="utf-8") if l.strip()]
    emb = {s: [json.loads(l) for l in (D / f"embeddings_v{TAG}" / f"{s}_embeddings.jsonl").open(encoding="utf-8")]
           for s in ("train", "validation", "test")}
    order = {r["task_id"]: i for i, r in enumerate(rows)}
    X = np.zeros((len(rows), 384))
    for s in emb:
        for r in emb[s]:
            X[order[r["task_id"]]] = r["embedding"]
    g = np.array([r["source"] for r in rows]); sp = np.array([r["split"] for r in rows])
    out = {"dataset": f"complexity_dataset_v{TAG}.jsonl", "rows": len(rows),
           "sources": len(set(g)), "splits": dict(collections.Counter(sp.tolist()))}
    print(f"FINAL EVALUATION — v{VERSION} · {len(rows)} rows · {len(set(g))} sources")
    for dim in ("type", "complexity", "domain"):
        y = np.array([r[dim] for r in rows]); labs = sorted(set(y))
        tr, te = sp != "test", sp == "test"
        yp = clone(LR()).fit(X[tr], y[tr]).predict(X[te])
        T, P = [], []
        for s in sorted(set(g)):
            mk = g == s
            if mk.sum() < 5 or not set(y[mk]) <= set(y[~mk]):
                continue
            P += list(clone(LR()).fit(X[~mk], y[~mk]).predict(X[mk])); T += list(y[mk])
        rnd_f1 = f1_score(y[te], yp, average="macro", zero_division=0)
        grp_f1 = f1_score(T, P, average="macro", zero_division=0)
        pr, rc, f1, su = precision_recall_fscore_support(T, P, labels=labs, zero_division=0)
        print(f"\n  {dim.upper()}  test acc={accuracy_score(y[te], yp):.3f}  "
              f"RANDOM macroF1={rnd_f1:.3f}  GROUPED macroF1={grp_f1:.3f}")
        for i, l in enumerate(labs):
            print(f"    {l:<38}{pr[i]:>7.3f}{rc[i]:>7.3f}{f1[i]:>7.3f}{su[i]:>8d}")
        out[dim] = {"test_accuracy": float(accuracy_score(y[te], yp)),
                    "random_macro_f1": float(rnd_f1), "grouped_macro_f1": float(grp_f1),
                    "grouped_per_class_f1": {l: float(f1[i]) for i, l in enumerate(labs)}}
    y = np.array([r["complexity"] for r in rows]); t = np.array([r["type"] for r in rows])
    T, P, R = [], [], []
    for s in sorted(set(g)):
        mk = g == s
        if mk.sum() < 5 or not set(y[mk]) <= set(y[~mk]):
            continue
        trm = ~mk
        P += list(clone(LR()).fit(X[trm], y[trm]).predict(X[mk]))
        rule = {ty: collections.Counter(y[trm][t[trm] == ty]).most_common(1)[0][0] for ty in set(t[trm])}
        R += [rule.get(x, "Low") for x in t[mk]]; T += list(y[mk])
    mf = f1_score(T, P, average="macro", zero_division=0)
    rf = f1_score(T, R, average="macro", zero_division=0)
    print(f"\n  ACCEPTANCE GATE  model={mf:.3f}  TYPE-lookup rule={rf:.3f}  "
          f"margin={mf-rf:+.3f}  -> {'PASS' if mf > rf else 'FAIL'}")
    out["acceptance_gate"] = {"model_macro_f1": float(mf), "rule_macro_f1": float(rf),
                              "margin": float(mf - rf), "verdict": "PASS" if mf > rf else "FAIL"}
    (D / f"final_evaluation_v{TAG}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  -> final_evaluation_v{TAG}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
