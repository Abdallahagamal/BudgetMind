from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import fakeredis

from service.classification_service import ClassificationService
from streaming.redis_stream import (
    RedisStreamConfig,
    RedisStreamConsumer,
    RedisStreamPublisher,
)
from worker.classification_worker import ClassificationWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_worker")


def run_verification() -> bool:
    logger.info("Starting Sprint 2 Member 5 Verification...")

    # 1. Setup fakeredis client & shared stream configuration
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:budgetmind:classification-tasks",
        group_name="test:classification-workers",
        consumer_name="test:worker-1",
        block_ms=1000,
    )

    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)

    # 2. Initialize in-process ClassificationService (loads .pkl models)
    logger.info("Initializing ClassificationService (in-process)...")
    service = ClassificationService()

    # 3. Setup temporary output file for Task Profiles
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as tmp_file:
        output_path = Path(tmp_file.name)

    try:
        worker = ClassificationWorker(
            consumer=consumer,
            service=service,
            output_path=output_path,
        )

        # -------------------------------------------------------------
        # TEST A: Successful Pipeline Flow
        # -------------------------------------------------------------
        logger.info("--- TEST A: Valid Task End-to-End ---")

        # Load real test vector from Sprint 1
        repo_root = Path(__file__).resolve().parents[3]
        test_file = repo_root / "Sprint1" / "embeddings_output" / "test_embeddings.jsonl"
        with open(test_file, "r", encoding="utf-8") as f:
            sample_data = json.loads(f.readline())

        task_id = sample_data["task_id"]
        embedding = sample_data["embedding"]
        logger.info("Publishing task %s (embedding dim=%d) to stream...", task_id, len(embedding))

        msg_id = publisher.publish(task_id, embedding)
        logger.info("Published message_id=%s", msg_id)

        # Execute single batch on worker
        processed = worker.run_once()
        assert processed == 1, f"Expected 1 processed message, got {processed}"

        # Verify output file
        with open(output_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 1, f"Expected 1 profile in output file, got {len(lines)}"
        profile = lines[0]
        logger.info("Saved Task Profile: %s", profile)

        # Verify contract fields
        expected_fields = [
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
        ]
        for field in expected_fields:
            assert field in profile, f"Missing expected field in TaskProfile: {field}"

        assert profile["task_id"] == task_id
        assert profile["taxonomy_version"] == "1.0"
        assert profile["from_cache"] is False

        # Verify ACK occurred (pending messages should be 0)
        pending = consumer.pending()
        pending_count = pending[0]["pending"] if pending else 0
        assert pending_count == 0, f"Expected 0 pending messages after ACK, got {pending_count}"
        logger.info("Verified: Message was successfully ACKed.")

        # -------------------------------------------------------------
        # TEST B: Failure Handling (No False ACK)
        # -------------------------------------------------------------
        logger.info("--- TEST B: Invalid Embedding Failure Handling ---")

        # Publish a bad message with invalid embedding (wrong dimension: 5 dims instead of 384)
        bad_task_id = "T_BAD_001"
        bad_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        bad_msg_id = publisher.publish(bad_task_id, bad_embedding)
        logger.info("Published invalid task %s message_id=%s", bad_task_id, bad_msg_id)

        processed = worker.run_once()
        assert processed == 0, f"Expected 0 processed messages for invalid task, got {processed}"

        # Verify output file line count did NOT increase
        with open(output_path, "r", encoding="utf-8") as f:
            lines_after = [json.loads(line) for line in f if line.strip()]
        assert len(lines_after) == 1, "Output file must not contain failed task"

        # Verify message was NOT ACKed (should remain in pending list!)
        pending_after = consumer.pending()
        bad_pending_count = pending_after[0]["pending"] if pending_after else 0
        assert bad_pending_count == 1, (
            f"Expected 1 pending message (no false ACK on failure), got {bad_pending_count}"
        )
        logger.info("Verified: Failed task was NOT falsely ACKed and remains pending.")

        logger.info("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        return True

    finally:
        consumer.close()
        if output_path.exists():
            output_path.unlink()


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
