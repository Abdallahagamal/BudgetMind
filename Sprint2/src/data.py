from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS_DIR = REPO_ROOT / "Sprint1" / "embeddings_output"

SPLITS = ("train", "validation", "test")
DIMENSIONS = ("type", "complexity", "domain")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    labels: dict[str, np.ndarray]
    groups: np.ndarray
    task_ids: np.ndarray
    texts: np.ndarray

    def __len__(self) -> int:
        return len(self.task_ids)

    def y(self, dimension: str) -> np.ndarray:
        if dimension not in self.labels:
            raise KeyError(f"unknown dimension {dimension!r}, expected one of {DIMENSIONS}")
        return self.labels[dimension]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_dataset(records: Iterable[dict[str, Any]]) -> Dataset:
    records = list(records)
    if not records:
        raise ValueError("no records to build a dataset from")

    X = np.asarray([r["embedding"] for r in records], dtype=np.float64)
    if X.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"expected {EMBEDDING_DIM}-dim embeddings, got {X.shape[1]}")

    norms = np.linalg.norm(X, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            "embeddings are not L2-normalized; the classification cache's cosine "
            "similarity assumes unit vectors (handoff Decision 2)"
        )

    return Dataset(
        X=X,
        labels={d: np.asarray([r[d] for r in records]) for d in DIMENSIONS},
        groups=np.asarray([r["source"] for r in records]),
        task_ids=np.asarray([r["task_id"] for r in records]),
        texts=np.asarray([r.get("text", "") for r in records], dtype=object),
    )


def load_split(split: str, embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR) -> Dataset:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    return _to_dataset(_read_jsonl(embeddings_dir / f"{split}_embeddings.jsonl"))


def load_pooled(embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR) -> Dataset:
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        records.extend(_read_jsonl(embeddings_dir / f"{split}_embeddings.jsonl"))
    return _to_dataset(records)
