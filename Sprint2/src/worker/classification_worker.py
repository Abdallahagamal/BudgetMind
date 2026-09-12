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

from service.classification_service import (  # noqa: E402
    ClassificationFailedError,
    ClassificationService,
    ClassificationServiceError,
    InvalidEmbeddingError,
    InvalidTaskError,
)
from streaming.redis_stream import RedisStreamConsumer, StreamMessage  # noqa: E402

logger = logging.getLogger("budgetmind.worker")

REPO_ROOT = Path(__file__).resolve().parents[3]

# Temporary Sprint 2 output location, since the Routing Layer / real storage
# doesn't exist yet. One JSON object per line (JSONL) so a single worker can
# safely append without re-reading/re-writing the whole file each time.
DEFAULT_OUTPUT_PATH = Path(
    os.getenv(
        "BUDGETMIND_TASK_PROFILE_OUTPUT",
        str(REPO_ROOT / "Sprint2" / "results" / "task_profiles.jsonl"),
    )
)

# Known classification errors indicating an individual task could not be classified.
_CLASSIFICATION_ERRORS = (
    InvalidTaskError,
    InvalidEmbeddingError,
    ClassificationFailedError,
    ClassificationServiceError,
)


class ClassificationWorker:
    """Orchestrates: Redis Stream -> Classification Service -> output file.

    Owns no ML logic and no Redis wire-protocol logic. It directly calls
    the in-process ClassificationService implemented by Member 4 and the
    RedisStreamConsumer queue abstraction.
    """

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
        """Process a single batch of messages from the Redis Stream.

        Returns:
            int: Number of messages successfully processed and acknowledged.
        """
        processed_count = 0
        for message in self._consumer.read():
            if self._handle(message):
                processed_count += 1
        return processed_count

    def run_forever(self) -> None:
        """Main loop. `consumer.read()` performs one blocking poll per call
        and returns, so the `while True` here is what keeps the worker alive
        across polls."""
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
        """Executes the pipeline for a single message in strict order:
        READ -> PROCESS -> CLASSIFY -> SAVE TASK PROFILE -> ACK.

        ACK happens only after the task profile has been durably persisted.

        Returns:
            bool: True if the message was successfully classified, saved,
                  and ACKed; False otherwise.
        """
        # Step 1: In-process classification call
        try:
            profile = self._service.classify(message.task_id, message.embedding)
        except _CLASSIFICATION_ERRORS as exc:
            # Task could not be classified. Do NOT ack: the message stays
            # pending in Redis so it isn't falsely marked as completed.
            logger.error(
                "classification failed for task_id=%s message_id=%s: %s",
                message.task_id,
                message.message_id,
                exc,
            )
            return False
        except Exception as exc:  # Safeguard against any unexpected service error
            logger.exception(
                "unexpected error during classification of task_id=%s message_id=%s: %s",
                message.task_id,
                message.message_id,
                exc,
            )
            return False

        # Step 2: Persist Task Profile to temporary output file
        try:
            self._write_profile(profile)
        except OSError as exc:
            # Could not persist the result to disk. Do NOT ack either --
            # the task was classified but not saved, so it must not be lost.
            logger.error(
                "failed to write task profile for task_id=%s message_id=%s: %s",
                message.task_id,
                message.message_id,
                exc,
            )
            return False

        # Step 3: Acknowledge in Redis stream ONLY after successful save
        self._consumer.ack(message.message_id)
        logger.info(
            "processed task_id=%s message_id=%s -> acked",
            message.task_id,
            message.message_id,
        )
        return True

    def _write_profile(self, profile: Any) -> None:
        """Appends the serialized TaskProfile to the output file."""
        record = self._serialize_profile(profile)
        with open(self._output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def _serialize_profile(profile: Any) -> dict[str, Any]:
        """Convert a TaskProfile instance into a dictionary.

        Reuses the TaskProfile's existing `.to_dict()` method defined in
        `src/task_profile.py`, with fallback to `dataclasses.asdict()`.
        """
        if hasattr(profile, "to_dict") and callable(profile.to_dict):
            return profile.to_dict()
        if dataclasses.is_dataclass(profile):
            return dataclasses.asdict(profile)

        # Fallback to dictionary extraction of contract attributes if needed
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
