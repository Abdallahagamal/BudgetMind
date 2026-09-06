from __future__ import annotations

import argparse
import csv
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELED = REPO_ROOT / "Sprint1" / "Label" / "labeled_dataset.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "Sprint1" / "dataset"

REQUIRED_FIELDS = (
    "task_id",
    "task_text",
    "source",
    "date_collected",
    "type",
    "complexity",
    "domain",
)

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def normalize_space(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text) if len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [
            {k: (v or "").strip() if isinstance(v, str) else v for k, v in row.items()}
            for row in reader
        ]


def build_clusters(rows: list[dict[str, str]], near_dup_threshold: float) -> dict[str, str]:
    """Union-find style clustering: near-duplicate rows are forced into the
    same split so a model can never be trained on one variant and tested on
    another. Returns task_id -> cluster_id.
    """
    ids = [r["task_id"] for r in rows]
    tokens = {r["task_id"]: tokenize(r["task_text"]) for r in rows}
    parent = {tid: tid for tid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    items = list(tokens.items())
    for i, (id_a, toks_a) in enumerate(items):
        for id_b, toks_b in items[i + 1 :]:
            if jaccard(toks_a, toks_b) >= near_dup_threshold:
                union(id_a, id_b)

    return {tid: find(tid) for tid in ids}


def stratified_group_split(
    rows: list[dict[str, str]],
    clusters: dict[str, str],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, str]:
    """Assigns each row to train/validation/test. Whole clusters move
    together; allocation is balanced per TYPE using cluster count as the
    unit (not row count), which keeps near-duplicate-heavy types like
    Architecture & System Design from skewing a single split.
    """
    rng = random.Random(seed)
    by_id = {r["task_id"]: r for r in rows}

    # group rows by (type, cluster)
    cluster_ids_by_type: dict[str, list[str]] = defaultdict(list)
    seen_clusters: set[str] = set()
    for r in rows:
        cid = clusters[r["task_id"]]
        if cid not in seen_clusters:
            seen_clusters.add(cid)
            cluster_ids_by_type[r["type"]].append(cid)

    cluster_rows: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        cluster_rows[clusters[r["task_id"]]].append(r["task_id"])

    assignment: dict[str, str] = {}
    for t, cids in cluster_ids_by_type.items():
        cids = list(cids)
        rng.shuffle(cids)
        n_rows = sum(len(cluster_rows[c]) for c in cids)
        target_train = round(n_rows * train_frac)
        target_val = round(n_rows * val_frac)

        used_train = used_val = 0
        for cid in cids:
            size = len(cluster_rows[cid])
            if used_train + size <= target_train or used_train == 0:
                split = "train"
                used_train += size
            elif used_val + size <= target_val or used_val == 0:
                split = "validation"
                used_val += size
            else:
                split = "test"
            for tid in cluster_rows[cid]:
                assignment[tid] = split

    return assignment


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(REQUIRED_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in REQUIRED_FIELDS})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split BudgetMind labeled dataset into train/val/test.")
    p.add_argument("--labeled", type=Path, default=DEFAULT_LABELED)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--near-dup-threshold", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_csv(args.labeled)

    missing = [r["task_id"] for r in rows if not r.get("type") or not r.get("complexity") or not r.get("domain")]
    if missing:
        print(f"ERROR: {len(missing)} rows still missing labels, fix these first: {missing}")
        return 1

    clusters = build_clusters(rows, args.near_dup_threshold)
    assignment = stratified_group_split(rows, clusters, args.train_frac, args.val_frac, args.seed)

    splits: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    for r in rows:
        splits[assignment[r["task_id"]]].append(r)

    write_split(args.out_dir / "train.csv", splits["train"])
    write_split(args.out_dir / "validation.csv", splits["validation"])
    write_split(args.out_dir / "test.csv", splits["test"])

    print(f"Total rows: {len(rows)}  (train={len(splits['train'])}, "
          f"validation={len(splits['validation'])}, test={len(splits['test'])})")

    print("\nPer-TYPE counts:")
    types = sorted({r["type"] for r in rows})
    for t in types:
        counts = {s: sum(1 for r in splits[s] if r["type"] == t) for s in splits}
        print(f"  {t:35s} train={counts['train']:>3}  val={counts['validation']:>3}  test={counts['test']:>3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())