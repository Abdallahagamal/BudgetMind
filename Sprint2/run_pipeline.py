from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from pathlib import Path

# Add Sprint2/src to sys.path so imports work regardless of launch directory
_CURRENT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _CURRENT_DIR / "src"

for _p in (str(_SRC_DIR), str(_CURRENT_DIR), str(_CURRENT_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPO_ROOT = _CURRENT_DIR.parent

import redis 

try:
    from service.classification_service import ClassificationService  
    from streaming.redis_stream import (  
        RedisStreamConfig,
        RedisStreamConsumer,
        RedisStreamPublisher,
    )
    from worker.classification_worker import ClassificationWorker, DEFAULT_OUTPUT_PATH   
except ImportError:
    try:
        from src.service.classification_service import ClassificationService   
        from src.streaming.redis_stream import (   
            RedisStreamConfig,
            RedisStreamConsumer,
            RedisStreamPublisher,
        )
        from src.worker.classification_worker import ClassificationWorker, DEFAULT_OUTPUT_PATH   
    except ImportError:
        from Sprint2.src.service.classification_service import ClassificationService   
        from Sprint2.src.streaming.redis_stream import (   
            RedisStreamConfig,
            RedisStreamConsumer,
            RedisStreamPublisher,
        )
        from Sprint2.src.worker.classification_worker import ClassificationWorker, DEFAULT_OUTPUT_PATH   

logger = logging.getLogger("budgetmind.pipeline")


def get_default_embeddings_dir() -> Path:
    """Locate the default embeddings directory, prioritizing embeddings_output_testing."""
    cand1 = REPO_ROOT / "Sprint1" / "budgetmind_embeddings" / "embeddings_output_testing"
    if cand1.exists():
        return cand1
    cand2 = REPO_ROOT / "Sprint1" / "embeddings_output"
    if cand2.exists():
        return cand2
    return cand1


def is_redis_available(host: str, port: int) -> bool:
    """Quickly probe if Redis port is open without hanging on retries."""
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def get_redis_client(config: RedisStreamConfig, force_fake: bool = False) -> redis.Redis:
    """Connect to live Redis if running; otherwise fall back to fakeredis."""
    if not force_fake and is_redis_available(config.host, config.port):
        try:
            client = redis.Redis(
                host=config.host,
                port=config.port,
                db=config.db,
                password=config.password,
                decode_responses=True,
                socket_connect_timeout=1.0,
            )
            client.ping()
            logger.info("Connected to live Redis server at %s:%s", config.host, config.port)
            return client
        except Exception as exc:
            logger.warning("Failed connecting to live Redis (%s). Using fakeredis.", exc)

    import fakeredis

    logger.info("Using in-memory Redis stream (fakeredis).")
    return fakeredis.FakeRedis(decode_responses=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BudgetMind Sprint 2 Pipeline: Ingest embeddings into Redis Stream and run Classification Worker"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=get_default_embeddings_dir(),
        help=f"Directory containing embeddings .jsonl files (default: {get_default_embeddings_dir()})",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,validation",
        help="Comma-separated list of splits to process (e.g., 'train,validation' or 'train,validation,test')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSONL file for Task Profiles (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear existing output file before running",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Force using in-memory fakeredis even if live Redis is available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    logger.info("Starting BudgetMind Sprint 2 Pipeline")
    logger.info("Embeddings directory: %s", args.embeddings_dir)
    logger.info("Splits to process: %s", splits)
    logger.info("Output file: %s", args.output)

    if args.clear_output and args.output.exists():
        args.output.unlink()
        logger.info("Cleared previous output file: %s", args.output)

    # 1. Setup Redis Queue with fast block timeout for batch pipeline
    stream_config = RedisStreamConfig(block_ms=1000)
    redis_client = get_redis_client(stream_config, force_fake=args.fake)

    publisher = RedisStreamPublisher(config=stream_config, client=redis_client)
    consumer = RedisStreamConsumer(config=stream_config, client=redis_client)

    # 2. Ingest embeddings and publish to Redis Stream
    total_published = 0
    split_counts: dict[str, int] = {}

    for split in splits:
        split_file = args.embeddings_dir / f"{split}_embeddings.jsonl"
        if not split_file.exists():
            logger.warning("Split file not found: %s (skipping)", split_file)
            continue

        count = 0
        with open(split_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    task_id = data.get("task_id")
                    embedding = data.get("embedding")
                    if not task_id or embedding is None:
                        logger.warning("Line %d in %s missing task_id or embedding", line_no, split_file.name)
                        continue

                    publisher.publish(task_id, embedding)
                    count += 1
                except Exception as exc:
                    logger.error("Failed to parse/publish line %d in %s: %s", line_no, split_file.name, exc)

        split_counts[split] = count
        total_published += count
        logger.info("Published %d tasks from %s to Redis Stream", count, split_file.name)

    if total_published == 0:
        logger.warning("No tasks were published to the stream. Exiting.")
        consumer.close()
        return

    # 3. Initialize Classification Service & Worker
    logger.info("Initializing ClassificationService (in-process)...")
    service = ClassificationService()

    worker = ClassificationWorker(
        consumer=consumer,
        service=service,
        output_path=args.output,
    )

    # 4. Drain the queue and process all published tasks
    logger.info("Worker processing %d published task(s)...", total_published)
    total_processed = 0

    while total_processed < total_published:
        processed_in_batch = worker.run_once()
        if processed_in_batch == 0:
            break
        total_processed += processed_in_batch

    consumer.close()

    # 5. Pipeline Summary
    print("\n" + "=" * 65)
    print("           BUDGETMIND SPRINT 2 PIPELINE SUMMARY           ")
    print("=" * 65)
    for split, count in split_counts.items():
        print(f"  - Split '{split}': {count} tasks published to queue")
    print(f"  - Total tasks published:  {total_published}")
    print(f"  - Total tasks classified: {total_processed}")
    print(f"  - Task profiles saved to: {args.output}")
    print("=" * 65)

    # Display sample profiles from output
    if args.output.exists():
        with open(args.output, "r", encoding="utf-8") as f:
            all_lines = [json.loads(line) for line in f if line.strip()]
        print(f"\nTotal records in {args.output.name}: {len(all_lines)}")
        print("\nLast 3 Task Profiles generated:")
        for prof in all_lines[-3:]:
            print(f"  Task {prof.get('task_id')}: Type={prof.get('type')} (conf={prof.get('type_confidence')}), "
                  f"Complexity={prof.get('complexity')}, Domain={prof.get('domain')}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
