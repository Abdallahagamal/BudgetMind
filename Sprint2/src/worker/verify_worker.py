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
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:budgetmind:classification-tasks",
        group_name="test:classification-workers",
        consumer_name="test:worker-1",
        block_ms=1000,
    )

    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)

    service = ClassificationService()

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as tmp_file:
        output_path = Path(tmp_file.name)

    try:
        worker = ClassificationWorker(
            consumer=consumer,
            service=service,
            output_path=output_path,
        )

        repo_root = Path(__file__).resolve().parents[3]
        test_file = repo_root / "Sprint1" / "embeddings_output" / "test_embeddings.jsonl"
        with open(test_file, "r", encoding="utf-8") as f:
            sample_data = json.loads(f.readline())

        task_id = sample_data["task_id"]
        embedding = sample_data["embedding"]

        msg_id = publisher.publish(task_id, embedding)

        processed = worker.run_once()
        assert processed == 1, f"Expected 1 processed message, got {processed}"

        with open(output_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 1, f"Expected 1 profile in output file, got {len(lines)}"
        profile = lines[0]

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

        pending = consumer.pending()
        pending_count = pending[0]["pending"] if pending else 0
        assert pending_count == 0, f"Expected 0 pending messages after ACK, got {pending_count}"

        bad_task_id = "T_BAD_001"
        bad_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        bad_msg_id = publisher.publish(bad_task_id, bad_embedding)

        processed = worker.run_once()
        assert processed == 0, f"Expected 0 processed messages for invalid task, got {processed}"

        with open(output_path, "r", encoding="utf-8") as f:
            lines_after = [json.loads(line) for line in f if line.strip()]
        assert len(lines_after) == 1, "Output file must not contain failed task"

        pending_after = consumer.pending()
        bad_pending_count = pending_after[0]["pending"] if pending_after else 0
        assert bad_pending_count == 1, (
            f"Expected 1 pending message (no false ACK on failure), got {bad_pending_count}"
        )

        return True

    finally:
        consumer.close()
        if output_path.exists():
            output_path.unlink()


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
