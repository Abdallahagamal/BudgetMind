from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import joblib  

from data import EMBEDDING_DIM  
from task_profile import (  
    ProbabilisticClassifier,
    TaskProfile,
    build_task_profile,
)

logger = logging.getLogger("budgetmind.service")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = REPO_ROOT / "Sprint2" / "models"

_MODEL_FILENAMES = {
    "type": "type_classifier_baseline.pkl",
    "complexity": "complexity_classifier_baseline.pkl",
    "domain": "domain_classifier_baseline.pkl",
}


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class ClassificationServiceError(Exception):
    """Base class for every error this service raises."""


class InvalidTaskError(ClassificationServiceError):
    """``task_id`` is missing, empty, or not a string."""


class InvalidEmbeddingError(ClassificationServiceError):
    """Embedding is missing, wrong shape, or contains non-finite values."""


class ModelLoadError(ClassificationServiceError):
    """A required ``.pkl`` model could not be found or unpickled."""


class ClassificationFailedError(ClassificationServiceError):
    """A loaded classifier raised while predicting on a valid input."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ClassificationServiceConfig:


    model_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("BUDGETMIND_MODEL_DIR", str(DEFAULT_MODEL_DIR))
        )
    )

    def __post_init__(self) -> None:
        # Accept a plain string too (e.g. a CLI arg or env var passed
        # straight through by a caller) without forcing every call site
        # to remember to wrap it in Path(...).
        if not isinstance(self.model_dir, Path):
            object.__setattr__(self, "model_dir", Path(self.model_dir))


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LoadedModels:
    type_model: ProbabilisticClassifier
    complexity_model: ProbabilisticClassifier
    domain_model: ProbabilisticClassifier


def load_models(model_dir: Path = DEFAULT_MODEL_DIR) -> LoadedModels:

    loaded: dict[str, ProbabilisticClassifier] = {}
    for dimension, filename in _MODEL_FILENAMES.items():
        path = model_dir / filename
        if not path.exists():
            raise ModelLoadError(
                f"missing {dimension} classifier at {path} "
                f"(expected one of the finalized Sprint 2 models)"
            )
        try:
            loaded[dimension] = joblib.load(path)
        except Exception as exc:  # pragma: no cover - depends on pickle internals
            raise ModelLoadError(f"failed to load {dimension} classifier from {path}: {exc}") from exc

    logger.info("loaded classifiers from %s", model_dir)
    return LoadedModels(
        type_model=loaded["type"],
        complexity_model=loaded["complexity"],
        domain_model=loaded["domain"],
    )


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #

class ClassificationService:

    def __init__(
        self,
        config: ClassificationServiceConfig | None = None,
        *,
        models: LoadedModels | None = None,
    ) -> None:
        self._config = config or ClassificationServiceConfig()
        self._models = models if models is not None else load_models(self._config.model_dir)

    def classify(self, task_id: str, embedding: Sequence[float]) -> TaskProfile:

        self._validate_task_id(task_id)
        vector = self._validate_embedding(embedding)

        try:
            return build_task_profile(
                task_id,
                vector,
                type_model=self._models.type_model,
                complexity_model=self._models.complexity_model,
                domain_model=self._models.domain_model,
            )
        except (InvalidTaskError, InvalidEmbeddingError):
            raise
        except Exception as exc:
            raise ClassificationFailedError(
                f"classification failed for task_id={task_id!r}: {exc}"
            ) from exc

    @staticmethod
    def _validate_task_id(task_id: str) -> None:
        if not isinstance(task_id, str) or not task_id.strip():
            raise InvalidTaskError(f"task_id must be a non-empty string, got {task_id!r}")

    @staticmethod
    def _validate_embedding(embedding: Sequence[float] | None) -> np.ndarray:
        if embedding is None:
            raise InvalidEmbeddingError("embedding is missing")

        try:
            vector = np.asarray(embedding, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidEmbeddingError(f"embedding could not be parsed as floats: {exc}") from exc

        if vector.ndim != 1 or vector.shape[0] != EMBEDDING_DIM:
            raise InvalidEmbeddingError(
                f"expected a {EMBEDDING_DIM}-dim embedding, got shape {vector.shape}"
            )

        if not np.all(np.isfinite(vector)):
            raise InvalidEmbeddingError("embedding contains NaN or infinite values")

        return vector
