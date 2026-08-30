"""
Basic sanity test for the embedding pipeline ONLY.

This does NOT build or evaluate a classifier. It only verifies that the
embedding generation step itself works correctly:

  1. The Sentence Transformer loads.
  2. Task text encodes successfully.
  3. Embedding dimension matches the model's advertised dimension.
  4. No NaN/null values are produced.
  5. Embedding count matches input record count.
  6. Task IDs stay aligned with their embeddings.
  7. Labels stay correctly associated with their original tasks.
  8. A rough cosine-similarity sanity check: semantically similar task
     descriptions should be closer to each other than to unrelated ones.

Run with:  python3 -m pytest tests/test_embedding_pipeline.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_embeddings import EmbeddingConfig, EmbeddingPipeline, load_jsonl  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "sample_data"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_pipeline_end_to_end(tmp_path):
    cfg = EmbeddingConfig()
    pipeline = EmbeddingPipeline(cfg)

    input_path = DATA_DIR / "train.jsonl"
    output_path = tmp_path / "train_embeddings.jsonl"

    result = pipeline.run_split(input_path, output_path, "train")

    original_records = load_jsonl(input_path)

    # 3 & 5: dimension + count
    assert result["status"] == "ok"
    assert result["count"] == len(original_records)
    assert result["embedding_dim"] == 384  # all-MiniLM-L6-v2 output size

    # Reload saved output
    with output_path.open() as f:
        saved = [json.loads(line) for line in f]

    assert len(saved) == len(original_records)

    orig_by_id = {r["task_id"]: r for r in original_records}

    for rec in saved:
        # 6: task IDs aligned
        assert rec["task_id"] in orig_by_id
        # 7: labels preserved correctly for the SAME task
        original = orig_by_id[rec["task_id"]]
        assert rec["type"] == original["type"]
        assert rec["complexity"] == original["complexity"]
        assert rec["domain"] == original["domain"]
        assert rec["text"] == original["text"]

        # 4: no NaNs
        vec = np.array(rec["embedding"])
        assert not np.isnan(vec).any()
        assert vec.shape[0] == 384

    print(f"\n[OK] {len(saved)} embeddings generated, all IDs/labels aligned, dim=384, no NaNs.")


def test_cosine_similarity_sanity(tmp_path):
    """Semantically related tasks should be more similar than unrelated ones."""
    cfg = EmbeddingConfig()
    pipeline = EmbeddingPipeline(cfg)

    texts = [
        "Fix authentication error in mobile login",       # 0: debugging/auth
        "Debug a memory leak in a Node.js background worker",  # 1: debugging (similar-ish to 0)
        "Draft a marketing email for a new product launch",    # 2: unrelated (marketing)
    ]
    embeddings = pipeline.embed_texts(texts)

    sim_related = cosine_sim(embeddings[0], embeddings[1])
    sim_unrelated = cosine_sim(embeddings[0], embeddings[2])

    print(f"\n[OK] cosine(debug, debug) = {sim_related:.3f} "
          f"vs cosine(debug, marketing) = {sim_unrelated:.3f}")

    assert sim_related > sim_unrelated, (
        "Expected two debugging-related task descriptions to be more "
        "semantically similar than a debugging task vs. a marketing task."
    )


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_pipeline_end_to_end(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_cosine_similarity_sanity(Path(td))
    print("\nAll sanity checks passed.")
