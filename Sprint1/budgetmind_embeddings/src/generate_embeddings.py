from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("budgetmind.embeddings")


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    id_field: str = "task_id"
    text_field: str = "text"
    batch_size: int = 64
    normalize_embeddings: bool = True



def load_jsonl(path: Path) -> list[dict[str, Any]]:

    records: list[dict[str, Any]] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                malformed += 1
                logger.warning(
                    "REPORT ONLY (not fixed): malformed JSON at %s line %d: %s",
                    path, line_no, e,
                )
    if malformed:
        logger.warning(
            "%d malformed row(s) found in %s and skipped. "
            "This is reported, not corrected — dataset integrity is Members 1-4's scope.",
            malformed, path,
        )
    return records


def validate_records(
    records: list[dict[str, Any]], cfg: EmbeddingConfig, split_name: str
) -> list[dict[str, Any]]:

    valid: list[dict[str, Any]] = []
    dropped = 0
    for rec in records:
        if cfg.id_field not in rec:
            dropped += 1
            logger.warning("Dropping record missing '%s' field: %s", cfg.id_field, rec)
            continue
        text = rec.get(cfg.text_field)
        if not isinstance(text, str) or not text.strip():
            dropped += 1
            logger.warning(
                "Dropping record %s: missing/empty '%s' field",
                rec.get(cfg.id_field, "<unknown>"), cfg.text_field,
            )
            continue
        valid.append(rec)

    if dropped:
        logger.warning(
            "[%s] %d record(s) could not be embedded and were reported above. "
            "%d/%d records will be embedded.",
            split_name, dropped, len(valid), len(records),
        )
    return valid


def save_embeddings(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    out_path: Path,
    cfg: EmbeddingConfig,
) -> None:

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec, vec in zip(records, embeddings):
            out_rec = dict(rec) 
            out_rec["embedding"] = vec.tolist()
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    npy_path = out_path.with_suffix(".npy")
    np.save(npy_path, embeddings)

    logger.info("Saved %d embeddings -> %s (+ %s)", len(records), out_path, npy_path.name)


class EmbeddingPipeline:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading Sentence Transformer: %s", self.cfg.model_name)
            t0 = time.time()
            self._model = SentenceTransformer(self.cfg.model_name)
            logger.info("Model loaded in %.2fs", time.time() - t0)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize_embeddings,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
        )
        return embeddings

    def run_split(
        self, input_path: Path, output_path: Path, split_name: str
    ) -> dict[str, Any]:
        logger.info("Processing split '%s' from %s", split_name, input_path)
        raw_records = load_jsonl(input_path)
        records = validate_records(raw_records, self.cfg, split_name)

        if not records:
            logger.error("No valid records to embed in split '%s'. Skipping.", split_name)
            return {"split": split_name, "count": 0, "status": "skipped_no_valid_records"}

        texts = [r[self.cfg.text_field] for r in records]
        embeddings = self.embed_texts(texts)

        assert embeddings.shape[0] == len(records), "embedding/record count mismatch"
        assert not np.isnan(embeddings).any(), "NaN values found in embeddings"

        save_embeddings(records, embeddings, output_path, self.cfg)

        return {
            "split": split_name,
            "count": len(records),
            "embedding_dim": int(embeddings.shape[1]),
            "status": "ok",
        }



def _default_data_dir() -> Path:
    if Path("sample_data").exists():
        return Path("sample_data")
    cand = Path(__file__).resolve().parent / "budgetmind_embeddings" / "sample_data"
    if cand.exists():
        return cand
    cand_parent = Path(__file__).resolve().parents[1] / "sample_data"
    if cand_parent.exists():
        return cand_parent
    return Path("sample_data")


def _default_output_dir() -> Path:
    if Path("sample_data").exists() or Path("embeddings_output").exists():
        return Path("embeddings_output")
    cand = Path(__file__).resolve().parent / "budgetmind_embeddings" / "embeddings_output"
    if cand.parent.exists():
        return cand
    cand_parent = Path(__file__).resolve().parents[1] / "embeddings_output"
    if cand_parent.parent.exists():
        return cand_parent
    return Path("embeddings_output")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate task embeddings from an existing labeled/split dataset "
                    "(BudgetMind Week 1, Member 5 scope only)."
    )
    p.add_argument(
        "--data-dir", type=Path, default=_default_data_dir(),
        help="Directory containing train.jsonl / validation.jsonl / test.jsonl "
             "produced by Members 1-4. Defaults to the bundled sample fixtures.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=_default_output_dir(),
        help="Directory to write embedding artifact files to.",
    )
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--id-field", type=str, default="task_id")
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--splits", type=str, nargs="+", default=list(DEFAULT_SPLITS),
        help="Which split filenames (without .jsonl) to process.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = EmbeddingConfig(
        model_name=args.model_name,
        id_field=args.id_field,
        text_field=args.text_field,
        batch_size=args.batch_size,
    )
    pipeline = EmbeddingPipeline(cfg)

    summary = []
    for split in args.splits:
        input_path = args.data_dir / f"{split}.jsonl"
        if not input_path.exists():
            logger.error("Input file not found, skipping: %s", input_path)
            summary.append({"split": split, "count": 0, "status": "input_not_found"})
            continue
        output_path = args.output_dir / f"{split}_embeddings.jsonl"
        result = pipeline.run_split(input_path, output_path, split)
        summary.append(result)

    logger.info("=" * 60)
    logger.info("EMBEDDING GENERATION SUMMARY")
    for r in summary:
        logger.info("  %s", r)
    logger.info("=" * 60)

    ok = any(r["status"] == "ok" for r in summary)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
