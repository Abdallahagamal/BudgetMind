"""
BudgetMind — Task Classifier Demo

Serves a single-page UI where a task description is typed in, embedded with
the same sentence-transformer used in Sprint 1 (all-MiniLM-L6-v2), and run
through the three Sprint 2/3 classifiers (type, complexity, domain).

Run from the root of the BudgetMind repo checkout:

    cd BudgetMind
    python -m venv .venv && source .venv/bin/activate   # (or .venv\\Scripts\\activate on Windows)
    pip install -r demo/requirements.txt
    uvicorn demo.app:app --reload

Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("budgetmind.demo")

# --------------------------------------------------------------------------- #
# Paths — this file lives in <repo_root>/demo/app.py
# --------------------------------------------------------------------------- #

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent

TYPE_MODEL_PATH = REPO_ROOT / "Sprint3" / "models" / "type_classifier_refined.pkl"
COMPLEXITY_MODEL_PATH = REPO_ROOT / "Sprint2" / "models" / "complexity_classifier_baseline.pkl"
DOMAIN_MODEL_PATH = REPO_ROOT / "Sprint2" / "models" / "domain_classifier_baseline.pkl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --------------------------------------------------------------------------- #
# App state — models load once at startup, not per-request
# --------------------------------------------------------------------------- #

app = FastAPI(title="BudgetMind Task Classifier Demo")

_state: dict[str, Any] = {"embedder": None, "type_model": None, "complexity_model": None, "domain_model": None}


@app.on_event("startup")
def load_models() -> None:
    for label, path in (
        ("type", TYPE_MODEL_PATH),
        ("complexity", COMPLEXITY_MODEL_PATH),
        ("domain", DOMAIN_MODEL_PATH),
    ):
        if not path.exists():
            raise RuntimeError(
                f"missing {label} classifier at {path}. "
                f"Make sure the demo/ folder sits at the root of the BudgetMind checkout."
            )
        _state[f"{label}_model"] = joblib.load(path)
        logger.info("loaded %s classifier from %s", label, path)

    logger.info("loading embedding model %s (first run downloads it)...", EMBEDDING_MODEL_NAME)
    from sentence_transformers import SentenceTransformer

    _state["embedder"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("ready.")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

class ClassifyRequest(BaseModel):
    text: str


def predict_with_margin(model, embedding: np.ndarray) -> dict[str, Any]:
    X = embedding.reshape(1, -1)
    proba = model.predict_proba(X)[0]
    order = np.argsort(proba)[::-1]

    top = float(proba[order[0]])
    runner_up = float(proba[order[1]]) if len(order) > 1 else 0.0
    label = str(model.classes_[order[0]])

    ranked = [
        {"label": str(model.classes_[i]), "probability": round(float(proba[i]), 4)}
        for i in order
    ]

    return {
        "label": label,
        "confidence": round(top, 4),
        "margin": round(top - runner_up, 4),
        "ranked": ranked,
    }


@app.post("/classify")
def classify(req: ClassifyRequest) -> JSONResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    embedder = _state["embedder"]
    embedding = embedder.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    if embedding.shape[0] != EMBEDDING_DIM:
        raise HTTPException(
            status_code=500,
            detail=f"expected a {EMBEDDING_DIM}-dim embedding, got shape {embedding.shape}",
        )

    result = {
        "task_id": "demo-request",
        "type": predict_with_margin(_state["type_model"], embedding),
        "complexity": predict_with_margin(_state["complexity_model"], embedding),
        "domain": predict_with_margin(_state["domain_model"], embedding),
    }
    return JSONResponse(result)


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #

@app.get("/")
def index() -> FileResponse:
    return FileResponse(DEMO_DIR / "static" / "index.html")
