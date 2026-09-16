"""Regenerate the v3.3 embeddings from the canonical dataset.

The embeddings are a generated artifact (~11 MB) and are gitignored. Run this once
after cloning, before training.

  python Sprint3/dataset_v3/src/make_embeddings.py
"""
from __future__ import annotations
import json, collections
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "Sprint3" / "dataset_v3" / "complexity_dataset_v3_3.jsonl"
OUT = ROOT / "Sprint3" / "dataset_v3" / "embeddings_v3_3"
MODEL = "all-MiniLM-L6-v2"


def main() -> int:
    rows = [json.loads(l) for l in DATASET.open(encoding="utf-8") if l.strip()]
    print(f"{len(rows)} rows from {DATASET.name}")
    X = SentenceTransformer(MODEL).encode(
        [r["text"] for r in rows], batch_size=64,
        normalize_embeddings=True, show_progress_bar=False)
    assert X.shape[1] == 384, X.shape
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-3)
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        idx = [i for i, r in enumerate(rows) if r["split"] == split]
        with (OUT / f"{split}_embeddings.jsonl").open("w", encoding="utf-8") as f:
            for i in idx:
                r = rows[i]
                f.write(json.dumps({
                    "task_id": r["task_id"], "text": r["text"], "source": r["source"],
                    "date_collected": r.get("collected_at") or "2026-09-16",
                    "type": r["type"], "complexity": r["complexity"], "domain": r["domain"],
                    "embedding": [float(v) for v in X[i]]}, ensure_ascii=False) + "\n")
        np.save(OUT / f"{split}_embeddings.npy", X[idx])
    print(f"wrote {dict(collections.Counter(r['split'] for r in rows))} to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
