from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from service.classification_service import (
    ClassificationFailedError,
    ClassificationService,
    ClassificationServiceError,
    InvalidEmbeddingError,
    InvalidTaskError,
)
from streaming.redis_stream import RedisStreamConsumer, StreamMessage

logger = logging.getLogger("budgetmind.worker")

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT_PATH = Path(
    os.getenv(
        "BUDGETMIND_TASK_PROFILE_OUTPUT",
        str(REPO_ROOT / "Sprint2" / "results" / "task_profiles.jsonl"),
    )
)

_CLASSIFICATION_ERRORS = (
    InvalidTaskError,
    InvalidEmbeddingError,
    ClassificationFailedError,
    ClassificationServiceError,
)


class ClassificationWorker:

    def __init__(
        self,
        consumer: RedisStreamConsumer,
        service: ClassificationService,
        output_path: Path | str = DEFAULT_OUTPUT_PATH,
    ) -> None:
        self._consumer = consumer
        self._service = service
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def run_once(self) -> int:
        processed_count = 0
        for message in self._consumer.read():
            if self._handle(message):
                processed_count += 1
        return processed_count

    def recover_pending(self, count: int | None = None) -> int:
        recovered_count = 0
        read_fn = getattr(self._consumer, "read_pending", None)
        if not callable(read_fn):
            return 0

        for message in read_fn(count=count):
            if self._handle(message):
                recovered_count += 1
        return recovered_count

    def run_forever(self) -> None:
        stream_name = getattr(self._consumer._config, "stream_name", "unknown")
        group_name = getattr(self._consumer._config, "group_name", "unknown")
        consumer_name = getattr(self._consumer._config, "consumer_name", "unknown")

        logger.info(
            "worker starting: stream=%s group=%s consumer=%s -> output=%s",
            stream_name,
            group_name,
            consumer_name,
            self._output_path,
        )
        try:
            while True:
                self.run_once()
        except KeyboardInterrupt:
            logger.info("shutdown requested, stopping worker loop")
        finally:
            self._consumer.close()

    def _handle(self, message: StreamMessage) -> bool:
        try:
            profile = self._service.classify(message.task_id, message.embedding)
        except _CLASSIFICATION_ERRORS as exc:
            logger.error(
                "classification failed for task_id=%s message_id=%s: %s",
                message.task_id,
                message.message_id,
                exc,
            )
            return False
        except Exception as exc:
            logger.exception(
                "unexpected error during classification of task_id=%s message_id=%s: %s",
                message.task_id,
                message.message_id,
                exc,
            )
            return False

        try:
            self._write_profile(profile)
        except OSError as exc:
            logger.error(
                "failed to write task profile for task_id=%s message_id=%s: %s",
                message.task_id,
                message.message_id,
                exc,
            )
            return False

        self._consumer.ack(message.message_id)
        logger.info(
            "processed task_id=%s message_id=%s -> acked",
            message.task_id,
            message.message_id,
        )
        return True

    def _write_profile(self, profile: Any) -> None:
        record = self._serialize_profile(profile)
        with open(self._output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def _serialize_profile(profile: Any) -> dict[str, Any]:
        if hasattr(profile, "to_dict") and callable(profile.to_dict):
            return profile.to_dict()
        if dataclasses.is_dataclass(profile):
            return dataclasses.asdict(profile)

        contract_fields = (
            "task_id",
            "taxonomy_version",
            "type",
            "complexity",
            "domain",
            "type_confidence",
            "complexity_confidence",
            "domain_confidence",
            "type_margin",
            "complexity_margin",
            "domain_margin",
            "embedding_ref",
            "classified_at",
            "from_cache",
        )
        return {f: getattr(profile, f) for f in contract_fields if hasattr(profile, f)}
